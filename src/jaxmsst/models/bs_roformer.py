from typing import Any, Sequence, Tuple

from einops import einsum, rearrange, pack, unpack, repeat
from flax import nnx
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def _normalize_rngs(rngs):
    if rngs is None:
        return None
    if isinstance(rngs, (nnx.Rngs, nnx.RngStream)):
        return rngs
    return nnx.Rngs(dropout=rngs)


class RotaryEmbedding(nnx.Module):
    """RoPE."""

    def __init__(
        self,
        min_timescale: int,
        max_timescale: int,
        embedding_dims: int = 0,
        cast_as_fprop_dtype: bool = True,
        fprop_dtype: jnp.dtype = jnp.bfloat16,
    ) -> None:
        if embedding_dims % 2:
            raise ValueError("Embedding dim for rotary position embedding must be a multiple of 2.")
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.embedding_dims = embedding_dims
        self.cast_as_fprop_dtype = cast_as_fprop_dtype
        self.fprop_dtype = fprop_dtype

    def __call__(self, inputs: jax.Array, position: jax.Array) -> jax.Array:
        assert position is not None
        if len(inputs.shape) != 4:
            raise ValueError("Input is assumed to be a rank 4 tensor of shape [batch, sequence, heads, dims].")
        if self.embedding_dims != inputs.shape[3]:
            raise ValueError(
                "The embedding dims of the rotary position embedding must match the hidden dimension of the inputs."
            )
        half_embedding_dim = self.embedding_dims // 2
        fraction = 2 * jnp.arange(0, half_embedding_dim) / self.embedding_dims
        timescale = self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction
        position = position[:, :, jnp.newaxis, jnp.newaxis]
        sinusoid_inp = position / timescale
        sin = jnp.sin(sinusoid_inp).astype(inputs.dtype)
        cos = jnp.cos(sinusoid_inp).astype(inputs.dtype)
        first_half, second_half = jnp.split(inputs, 2, axis=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        if self.cast_as_fprop_dtype:
            first_part = first_part.astype(self.fprop_dtype)
            second_part = second_part.astype(self.fprop_dtype)
        x_out = jnp.concatenate((first_part, second_part), axis=-1)
        return x_out


class RMSNorm(nnx.Module):
    """RMS normalization."""

    def __init__(
        self,
        dim: int,
        dtype: Any = jnp.float32,
        weight_dtype: Any = jnp.float32,
    ) -> None:
        self.dim = dim
        self.dtype = dtype
        self.weight_dtype = weight_dtype
        self.gamma = nnx.Param(jnp.ones((dim,), dtype=weight_dtype))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        gamma = jnp.asarray(self.gamma, self.dtype)
        y = x * gamma * (self.dim ** 0.5)
        return y


class FeedForward(nnx.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, *, rngs: nnx.Rngs) -> None:
        self.dim = dim
        self.mult = mult
        self.dropout = dropout
        dim_inner = int(dim * mult)
        self.RMSNorm_0 = RMSNorm(dim)
        self.Dense_0 = nnx.Linear(dim, dim_inner, rngs=rngs)
        self.Dropout_0 = nnx.Dropout(dropout)
        self.Dense_1 = nnx.Linear(dim_inner, dim, rngs=rngs)
        self.Dropout_1 = nnx.Dropout(dropout)

    def __call__(self, x: jnp.ndarray, deterministic: bool, rngs=None) -> jnp.ndarray:
        rngs = _normalize_rngs(rngs)
        x = self.RMSNorm_0(x)
        x = self.Dense_0(x)
        x = nnx.gelu(x)
        x = self.Dropout_0(x, deterministic=deterministic, rngs=rngs)
        x = self.Dense_1(x)
        x = self.Dropout_1(x, deterministic=deterministic, rngs=rngs)
        return x


class Attend(nnx.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        self.dropout = dropout

    def __call__(self, q, k, v, deterministic: bool) -> jnp.ndarray:
        scale = q.shape[-1] ** -0.5
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = jax.nn.dot_product_attention(q, k, v, scale=scale)
        return out.transpose(0, 2, 1, 3)


def get_seq_pos(seq_len, offset=0):
    return jnp.arange(seq_len) + offset


def embed_forward(t: jnp.ndarray, seq_len=None, offset=0, dim_head=None, freqs=None):
    freqs = einsum(t, freqs, "..., f -> ... f")
    freqs = repeat(freqs, "... n -> ... (n r)", r=2)
    return freqs


def rotate_queries_or_keys(t, seq_dim=None, offset=0, scale=None, dim_head=None, freqs=None):
    seq_dim = -2
    seq_len = t.shape[seq_dim]
    seq = get_seq_pos(seq_len, offset=offset)
    freqs = embed_forward(seq, seq_len=seq_len, offset=offset, dim_head=dim_head, freqs=freqs)
    if seq_dim == -3:
        freqs = rearrange(freqs, "n d -> n 1 d")
    return apply_rotary_emb(freqs, t, scale=default(scale, 1.0), seq_dim=seq_dim)


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = jax_unstack(x, axis=-1)
    x = jnp.stack((-x2, x1), axis=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * jnp.cos(freqs) * scale) + (rotate_half(t) * jnp.sin(freqs) * scale)
    out = jnp.concatenate((t_left, t, t_right), axis=-1)

    return out


class Attention(nnx.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        shared_qkv_bias: bool = False,
        shared_out_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.shared_qkv_bias = shared_qkv_bias
        self.shared_out_bias = shared_out_bias
        dim_inner = heads * dim_head
        self.RMSNorm_0 = RMSNorm(dim)
        self.to_qkv = nnx.Linear(dim, dim_inner * 3, use_bias=shared_qkv_bias, rngs=rngs)
        self.to_gates = nnx.Linear(dim, heads, rngs=rngs)
        self.to_out = nnx.Linear(dim_inner, dim, use_bias=shared_out_bias, rngs=rngs)
        self.Dropout_0 = nnx.Dropout(dropout)
        self.attend = Attend(dropout=dropout)
        self.freqs = nnx.Param(jnp.ones((dim_head // 2,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray, deterministic: bool, rngs=None) -> jnp.ndarray:
        rngs = _normalize_rngs(rngs)
        dim_inner = self.heads * self.dim_head
        x = self.RMSNorm_0(x)
        temp_x = self.to_qkv(x)
        q, k, v = rearrange(temp_x, "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
        freqs = self.freqs
        q = rotate_queries_or_keys(q, dim_head=self.dim_head, freqs=freqs)
        k = rotate_queries_or_keys(k, dim_head=self.dim_head, freqs=freqs)
        out = self.attend(q, k, v, deterministic=deterministic)
        gates = self.to_gates(x)
        out = out * nnx.sigmoid(rearrange(gates, "b n h -> b h n 1"))
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        out = self.Dropout_0(out, deterministic=deterministic, rngs=rngs)
        return out


class Transformer(nnx.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        ff_mult: int = 4,
        norm_output: bool = True,
        shared_qkv_bias: bool = False,
        shared_out_bias: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.depth = depth
        self.dim_head = dim_head
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.ff_mult = ff_mult
        self.norm_output = norm_output
        self.shared_qkv_bias = shared_qkv_bias
        self.shared_out_bias = shared_out_bias
        self.layers = []
        for layer_idx in range(depth):
            attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                shared_qkv_bias=shared_qkv_bias,
                shared_out_bias=shared_out_bias,
                rngs=rngs,
            )
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, rngs=rngs)
            attn_name = f"layers_{layer_idx}_0"
            ff_name = f"layers_{layer_idx}_1"
            setattr(self, attn_name, attn)
            setattr(self, ff_name, ff)
            self.layers.append((attn_name, ff_name))

    def __call__(self, x: jnp.ndarray, deterministic: bool, rngs=None) -> jnp.ndarray:
        for attn_name, ff_name in self.layers:
            attn = getattr(self, attn_name)
            ff = getattr(self, ff_name)
            x = attn(x, deterministic=deterministic, rngs=rngs) + x
            x = ff(x, deterministic=deterministic, rngs=rngs) + x
        return x


class BandSplit(nnx.Module):
    def __init__(
        self,
        dim: int,
        freqs_per_bands_with_complex: Sequence[int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.freqs_per_bands_with_complex = tuple(freqs_per_bands_with_complex)
        self.freqs_per_bands_with_complex_cum = list(np.cumsum(self.freqs_per_bands_with_complex))
        self._band_indices = []
        for idx, dim_in in enumerate(self.freqs_per_bands_with_complex):
            rms = RMSNorm(dim_in)
            dense = nnx.Linear(dim_in, dim, rngs=rngs)
            setattr(self, f"RMSNorm_{idx}", rms)
            setattr(self, f"Dense_{idx}", dense)
            self._band_indices.append(idx)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.split(x, self.freqs_per_bands_with_complex_cum, axis=-1)
        outs = []
        for idx, split_input in zip(self._band_indices, x):
            rms = getattr(self, f"RMSNorm_{idx}")
            dense = getattr(self, f"Dense_{idx}")
            split_output = dense(rms(split_input))
            outs.append(split_output)
        return jnp.stack(outs, axis=-2)


def jax_unstack(x, axis=0):
    return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]


class MLP(nnx.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int = None,
        depth: int = 1,
        activation=nnx.tanh,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        dim_hidden = default(dim_hidden, dim_in)
        dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
        self.activation = activation
        self._layer_names = []
        for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            layer = nnx.Linear(layer_dim_in, layer_dim_out, rngs=rngs)
            layer_name = f"layers_{ind * 2}"
            setattr(self, layer_name, layer)
            self._layer_names.append(layer_name)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for idx, layer_name in enumerate(self._layer_names):
            layer = getattr(self, layer_name)
            x = layer(x)
            if idx < len(self._layer_names) - 1:
                x = self.activation(x)
        return x


class _ToFreqs(nnx.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        mlp_expansion_factor: int,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        dim_hidden = dim * mlp_expansion_factor
        self.layers_0 = MLP(dim, dim_out * 2, dim_hidden=dim_hidden, depth=depth, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.layers_0(x)
        return nnx.glu(x)


class MaskEstimator(nnx.Module):
    def __init__(
        self,
        dim: int,
        dim_inputs: Sequence[int],
        depth: int,
        mlp_expansion_factor: int = 4,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.dim_inputs = tuple(dim_inputs)
        self.depth = depth
        self.mlp_expansion_factor = mlp_expansion_factor
        self._to_freqs_names = []
        for idx, dim_in in enumerate(self.dim_inputs):
            to_freqs = _ToFreqs(
                dim=dim,
                dim_out=dim_in,
                depth=depth,
                mlp_expansion_factor=mlp_expansion_factor,
                rngs=rngs,
            )
            name = f"to_freqs_{idx}"
            setattr(self, name, to_freqs)
            self._to_freqs_names.append(name)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jax_unstack(x, axis=-2)
        outs = []
        for band_features, name in zip(x, self._to_freqs_names):
            mlp = getattr(self, name)
            freq_out = mlp(band_features)
            outs.append(freq_out)
        return jnp.concatenate(outs, axis=-1)


DEFAULT_FREQS_PER_BANDS = [
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
]


class BSRoformer(nnx.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 8,
        stereo: bool = True,
        num_stems: int = 1,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        linear_transformer_depth: int = 0,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        dim_freqs_in: int = 1025,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        stft_normalized: int = False,
        mask_estimator_depth: int = 2,
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
        use_shared_bias: bool = False,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        self.dim = dim
        self.depth = depth
        self.stereo = stereo
        self.num_stems = num_stems
        self.time_transformer_depth = time_transformer_depth
        self.freq_transformer_depth = freq_transformer_depth
        self.linear_transformer_depth = linear_transformer_depth
        self.dim_head = dim_head
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.dim_freqs_in = dim_freqs_in
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.stft_normalized = stft_normalized
        self.mask_estimator_depth = mask_estimator_depth
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_hop_size = multi_stft_hop_size
        self.multi_stft_normalized = multi_stft_normalized
        self.use_shared_bias = use_shared_bias

        if rngs is None:
            rngs = nnx.Rngs(0)

        freqs_per_bands_with_complex = tuple(freq * 2 * 2 for freq in DEFAULT_FREQS_PER_BANDS)
        self.freqs_per_bands_with_complex = freqs_per_bands_with_complex

        self.BandSplit_0 = BandSplit(
            dim=dim,
            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
            rngs=rngs,
        )

        self._time_transformer_names = []
        self._freq_transformer_names = []
        for idx in range(depth):
            time_transformer = Transformer(
                depth=time_transformer_depth,
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                norm_output=False,
                shared_qkv_bias=use_shared_bias,
                shared_out_bias=use_shared_bias,
                rngs=rngs,
            )
            freq_transformer = Transformer(
                depth=freq_transformer_depth,
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                attn_dropout=attn_dropout,
                ff_dropout=ff_dropout,
                norm_output=False,
                shared_qkv_bias=use_shared_bias,
                shared_out_bias=use_shared_bias,
                rngs=rngs,
            )
            time_name = f"time_transformer_{idx}"
            freq_name = f"freq_transformer_{idx}"
            setattr(self, time_name, time_transformer)
            setattr(self, freq_name, freq_transformer)
            self._time_transformer_names.append(time_name)
            self._freq_transformer_names.append(freq_name)

        self._mask_estimator_names = []
        for idx in range(num_stems):
            estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                rngs=rngs,
            )
            name = f"MaskEstimator_{idx}"
            setattr(self, name, estimator)
            self._mask_estimator_names.append(name)

        self.RMSNorm_0 = RMSNorm(dim)

    def __call__(self, raw_audio, deterministic: bool = False, rngs=None):
        rngs = _normalize_rngs(rngs)
        audio_channels = 2 if self.stereo else 1

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
            self.stereo and channels == 2
        ), (
            "stereo needs to be set to True if passing in audio signal that is stereo "
            "(channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")

        _, _, stft_repr = jax.scipy.signal.stft(
            raw_audio,
            nfft=self.stft_n_fft,
            noverlap=self.stft_win_length - self.stft_hop_length,
            nperseg=self.stft_win_length,
            boundary=None,
        )
        spectrum_win = jnp.sin(jnp.linspace(0, jnp.pi, self.stft_win_length, endpoint=False)) ** 2
        stft_repr *= spectrum_win.sum()
        stft_repr = as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        x = rearrange(stft_repr, "b f t c -> b t (f c)")

        x = self.BandSplit_0(x)

        for time_name, freq_name in zip(self._time_transformer_names, self._freq_transformer_names):
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")

            time_transformer = getattr(self, time_name)
            x = time_transformer(x, deterministic=deterministic, rngs=rngs)

            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")

            freq_transformer = getattr(self, freq_name)
            x = freq_transformer(x, deterministic=deterministic, rngs=rngs)

            (x,) = unpack(x, ps, "* f d")

        x = self.RMSNorm_0(x)
        out = []
        for name in self._mask_estimator_names:
            estimator = getattr(self, name)
            res = estimator(x)
            out.append(res)
        mask = jnp.stack(out, axis=1)
        mask = rearrange(mask, "b n t (f c) -> b n f t c", c=2)
        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        stft_repr = as_complex(stft_repr)
        mask = as_complex(mask)

        stft_repr = stft_repr * mask

        stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=audio_channels)
        _, recon_audio = jax.scipy.signal.istft(
            stft_repr,
            nfft=self.stft_n_fft,
            noverlap=self.stft_win_length - self.stft_hop_length,
            nperseg=self.stft_win_length,
            boundary=False,
            input_onesided=True,
        )
        recon_audio /= spectrum_win.sum()

        recon_audio = rearrange(recon_audio, "(b n s) t -> b n s t", s=audio_channels, n=self.num_stems)
        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        return recon_audio


def as_complex(x):
    assert x.shape[-1] == 2
    return jax.lax.complex(x[..., 0], x[..., 1])


def as_real(x):
    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        return x

    xr = jnp.zeros(x.shape + (2,), dtype=x.real.dtype)
    xr = xr.at[..., 0].set(x.real)
    xr = xr.at[..., 1].set(x.imag)
    return xr


if __name__ == "__main__":
    test = BSRoformer(dim=256, depth=1, rngs=nnx.Rngs(0))
    output = test(jnp.ones((1, 2, 16000)), deterministic=True)
    print(output.shape)
