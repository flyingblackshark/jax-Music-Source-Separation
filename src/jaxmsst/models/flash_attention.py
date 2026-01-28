from __future__ import annotations

from typing import Optional

import jax


def apply_attention(
    query: jax.Array,
    key: jax.Array,
    value: jax.Array,
    scale: float,
    attention_type: str,
    flash_config: Optional[object] = None,
    min_seq_len: int = 0,
) -> jax.Array:
    del attention_type, flash_config, min_seq_len
    return jax.nn.dot_product_attention(query, key, value, scale=scale)
