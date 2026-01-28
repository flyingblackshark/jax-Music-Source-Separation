from __future__ import annotations

from dataclasses import dataclass
import functools
from typing import Optional

import jax
import jax.numpy as jnp


@dataclass
class FlashConfig:
    block_q: int = 128
    block_kv: int = 128


_FLASH_ATTN_MESH = None


def set_flash_attention_mesh(mesh) -> None:
    """Provide the Mesh used for shard_map wrapping of Mosaic kernels (TPU flash attention)."""
    global _FLASH_ATTN_MESH
    _FLASH_ATTN_MESH = mesh


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _pad_for_flash(tensor: jax.Array, block_size: int) -> tuple[jax.Array, int, int]:
    seq_len = tensor.shape[2]
    head_dim = tensor.shape[3]
    pad_seq = (-seq_len) % block_size
    pad_head = max(0, 128 - head_dim) if head_dim < 128 else 0
    if pad_seq or pad_head:
        tensor = jnp.pad(tensor, ((0, 0), (0, 0), (0, pad_seq), (0, pad_head)))
    return tensor, seq_len, head_dim


def _tokamax_flash_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    scale: float,
    config: FlashConfig,
) -> jax.Array:
    from jax.experimental.pallas.ops.tpu.splash_attention import splash_attention_kernel
    from tokamax._src.ops.experimental.tpu.splash_attention import (
        splash_attention_mask as tokamax_splash_attention_mask,
    )
    from tokamax._src.ops.experimental.tpu.splash_attention import (
        splash_attention_kernel as tokamax_splash_attention_kernel,
    )

    # Tokamax Splash enforces some block-size constraints (e.g. KV compute block multiple of 128).
    # We keep the blocks fixed and pad sequence length up to the block size.
    block_q = max(128, _round_up_to_multiple(int(config.block_q), 128))
    block_kv = max(128, _round_up_to_multiple(int(config.block_kv), 128))

    query, query_len, head_dim = _pad_for_flash(query, block_q)
    key, key_len, _ = _pad_for_flash(key * scale, block_kv)
    value, _, _ = _pad_for_flash(value, block_kv)

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_q,
        block_kv_compute=block_kv,
        block_kv=block_kv,
        block_q_dkv=block_q,
        block_kv_dkv=block_kv,
        block_kv_dkv_compute=block_kv,
        block_q_dq=None,
        block_kv_dq=None,
        use_fused_bwd_kernel=True,
    )

    splash_config = tokamax_splash_attention_kernel.SplashConfig(
        block_q=block_sizes.block_q,
        block_kv=block_sizes.block_kv,
        block_kv_compute=block_sizes.block_kv_compute,
        block_q_dkv=block_sizes.block_q_dkv,
        block_kv_dkv=block_sizes.block_kv_dkv,
        block_kv_dkv_compute=block_sizes.block_kv_dkv_compute,
        block_q_dq=None,
        block_kv_dq=None,
        use_fused_bwd_kernel=True,
        q_layout=tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
        k_layout=tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
        v_layout=tokamax_splash_attention_kernel.QKVLayout.HEAD_DIM_MINOR,
    )

    mask = tokamax_splash_attention_mask.FullMask(_shape=(query.shape[2], key.shape[2]))
    splash_kernel = tokamax_splash_attention_kernel.make_splash_mha(
        mask=mask,
        q_seq_shards=1,
        config=splash_config,
        save_residuals=False,
    )
    vmapped_splash = jax.vmap(splash_kernel, in_axes=(0, 0, 0, None))
    output = vmapped_splash(query, key, value, None)
    return output[:, :, :query_len, :head_dim]


def apply_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    scale: float,
    attention_type: str,
    flash_config: Optional[FlashConfig] = None,
    min_seq_len: int = 0,
) -> jax.Array:
    if attention_type in ("tokamax_attention", "tokamax_flash"):
        if min_seq_len and query.shape[1] < min_seq_len:
            return jax.nn.dot_product_attention(query, key, value, scale=scale)
        # Tokamax Splash attention is TPU-only; attempting to lower on CPU/GPU can hang or error.
        if jax.default_backend() != "tpu":
            return jax.nn.dot_product_attention(query, key, value, scale=scale)
        try:
            from jax.experimental import shard_map
            from jax.sharding import PartitionSpec

            query_t = query.transpose(0, 2, 1, 3)
            key_t = key.transpose(0, 2, 1, 3)
            value_t = value.transpose(0, 2, 1, 3)
            if _FLASH_ATTN_MESH is not None:
                # Mosaic kernels can't be auto-partitioned. shard_map makes the sharding explicit.
                in_spec = PartitionSpec("data", None, None, None)

                @functools.partial(
                    shard_map.shard_map,
                    mesh=_FLASH_ATTN_MESH,
                    in_specs=(in_spec, in_spec, in_spec),
                    out_specs=in_spec,
                    check_rep=False,
                )
                def _wrapped(q, k, v):
                    return _tokamax_flash_attention(q, k, v, scale, flash_config or FlashConfig())

                output = _wrapped(query_t, key_t, value_t)
            else:
                output = _tokamax_flash_attention(
                    query_t,
                    key_t,
                    value_t,
                    scale,
                    flash_config or FlashConfig(),
                )
            return output.transpose(0, 2, 1, 3)
        except Exception:
            pass
    return jax.nn.dot_product_attention(query, key, value, scale=scale)
