from functools import partial
from typing import Any, Sequence, Tuple

from einops import einsum, rearrange, pack, unpack, repeat, reduce
from flax import nnx
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
from librosa import filters

from .flash_attention import apply_attention

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
    def __init__(
        self,
        dropout: float = 0.0,
        attention_type: str = "dot_product",
        flash_min_seq_len: int = 0,
    ) -> None:
        self.dropout = dropout
        self.attention_type = attention_type
        self.flash_min_seq_len = flash_min_seq_len

    def __call__(self, q, k, v, deterministic: bool) -> jnp.ndarray:
        scale = q.shape[-1] ** -0.5
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        out = apply_attention(
            q,
            k,
            v,
            scale=scale,
            attention_type=self.attention_type,
            min_seq_len=self.flash_min_seq_len,
        )
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
        attention_type: str = "dot_product",
        flash_min_seq_len: int = 0,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        dim_inner = heads * dim_head
        self.RMSNorm_0 = RMSNorm(dim)
        self.to_qkv = nnx.Linear(dim, dim_inner * 3, use_bias=False, rngs=rngs)
        self.to_gates = nnx.Linear(dim, heads, rngs=rngs)
        self.to_out = nnx.Linear(dim_inner, dim, use_bias=False, rngs=rngs)
        self.Dropout_0 = nnx.Dropout(dropout)
        self.attend = Attend(
            dropout=dropout,
            attention_type=attention_type,
            flash_min_seq_len=flash_min_seq_len,
        )
        self.freqs = nnx.Param(jnp.ones((dim_head // 2,), dtype=jnp.float32))

    def __call__(self, x: jnp.ndarray, deterministic: bool, rngs=None) -> jnp.ndarray:
        rngs = _normalize_rngs(rngs)
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
        attention_type: str = "dot_product",
        flash_min_seq_len: int = 0,
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
        self.layers = []
        for layer_idx in range(depth):
            attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                attention_type=attention_type,
                flash_min_seq_len=flash_min_seq_len,
                rngs=rngs,
            )
            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, rngs=rngs)
            attn_name = f"layers_{layer_idx}_0"
            ff_name = f"layers_{layer_idx}_1"
            setattr(self, attn_name, attn)
            setattr(self, ff_name, ff)
            self.layers.append((attn_name, ff_name))
        self.norm = RMSNorm(dim)

    def __call__(self, x: jnp.ndarray, deterministic: bool, rngs=None) -> jnp.ndarray:
        for attn_name, ff_name in self.layers:
            attn = getattr(self, attn_name)
            ff = getattr(self, ff_name)
            x = attn(x, deterministic=deterministic, rngs=rngs) + x
            x = ff(x, deterministic=deterministic, rngs=rngs) + x
        return self.norm(x)


class BandSplit(nnx.Module):
    def __init__(
        self,
        dim: int,
        freqs_per_bands_with_complex: Sequence[int],
        freqs_per_bands_with_complex_cum: Sequence[int],
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.dim = dim
        self.freqs_per_bands_with_complex = tuple(freqs_per_bands_with_complex)
        self.freqs_per_bands_with_complex_cum = list(freqs_per_bands_with_complex_cum)
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


class MelBandRoformer(nnx.Module):
    def __init__(
        self,
        dim: int = 384,
        depth: int = 6,
        stereo: bool = True,
        num_stems: int = 1,
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        linear_transformer_depth: int = 0,
        num_bands: int = 60,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        attention_type: str = "dot_product",
        flash_min_seq_len: int = 0,
        dim_freqs_in: int = 1025,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 441,
        stft_win_length: int = 2048,
        stft_normalized: int = False,
        mask_estimator_depth: int = 3,
        multi_stft_resolution_loss_weight: float = 1.0,
        multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size: int = 147,
        multi_stft_normalized: bool = False,
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
        self.num_bands = num_bands
        self.dim_head = dim_head
        self.heads = heads
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.attention_type = attention_type
        self.flash_min_seq_len = flash_min_seq_len
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

        if rngs is None:
            rngs = nnx.Rngs(0)

        mel_filter_bank_numpy = filters.mel(sr=44100, n_fft=stft_n_fft, n_mels=num_bands)
        mel_filter_bank_numpy[0][0] = 1.0
        mel_filter_bank_numpy[-1, -1] = 1.0
        freqs_per_band = mel_filter_bank_numpy > 0
        freqs = stft_n_fft // 2 + 1
        repeated_freq_indices = repeat(np.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]
        freq_indices = repeat(freq_indices, "f -> f s", s=2)
        freq_indices = freq_indices * 2 + np.arange(2)
        freq_indices = rearrange(freq_indices, "f s -> (f s)")

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")
        freqs_per_bands_with_complex = tuple(2 * f * 2 for f in num_freqs_per_band.tolist())
        freqs_per_bands_with_complex_cum = np.cumsum(np.asarray(freqs_per_bands_with_complex))

        self.freq_indices = tuple(freq_indices.tolist())
        self.num_bands_per_freq = tuple(num_bands_per_freq.tolist())
        self.freqs_per_bands_with_complex = freqs_per_bands_with_complex
        self.freqs_per_bands_with_complex_cum = list(freqs_per_bands_with_complex_cum)

        self.BandSplit_0 = BandSplit(
            dim=dim,
            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
            freqs_per_bands_with_complex_cum=freqs_per_bands_with_complex_cum,
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
                attention_type=attention_type,
                flash_min_seq_len=flash_min_seq_len,
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
                attention_type=attention_type,
                flash_min_seq_len=flash_min_seq_len,
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

    def __call__(self, raw_audio, deterministic: bool = False, rngs=None):
        rngs = _normalize_rngs(rngs)
        audio_channels = 2 if self.stereo else 1

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        batch, channels, _ = raw_audio.shape

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

        freq_indices = jnp.asarray(self.freq_indices)
        num_bands_per_freq = jnp.asarray(self.num_bands_per_freq)

        batch_arange = jnp.arange(batch)[..., None]
        x = stft_repr[batch_arange, freq_indices]

        x = rearrange(x, "b f t c -> b t (f c)")

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

        out = []
        for name in self._mask_estimator_names:
            estimator = getattr(self, name)
            res = estimator(x)
            out.append(res)
        masks = jnp.stack(out, axis=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        stft_repr = as_complex(stft_repr)
        masks = as_complex(masks)

        scatter_indices = repeat(freq_indices, "f -> b n f t", b=batch, n=self.num_stems, t=stft_repr.shape[-1])
        stft_repr_expanded_stems = repeat(stft_repr, "b 1 ... -> b n ...", n=self.num_stems)

        masks_summed = scatter(
            input=jnp.zeros_like(stft_repr_expanded_stems),
            dim=2,
            index=scatter_indices,
            src=masks,
            reduce="add",
        )

        denom = repeat(num_bands_per_freq, "f -> (f r) 1", r=channels)
        masks_averaged = masks_summed / jnp.clip(denom, min=1e-8)

        stft_repr = stft_repr * masks_averaged
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


def scatter(input, dim, index, src, reduce=None):
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )

    if reduce is None:
        _scatter = jax.lax.scatter
    elif reduce == "add":
        _scatter = jax.lax.scatter_add
    elif reduce == "multiply":
        _scatter = jax.lax.scatter_mul

    _scatter = partial(_scatter, dimension_numbers=dnums)
    vmap_inner = partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)

    for _ in range(len(input.shape) - 1):
        _scatter = vmap_inner(_scatter)
    swap = lambda x: jnp.swapaxes(x, dim, -1)
    input, index, src = list(map(swap, (input, index, src)))
    return swap(_scatter(input, jnp.expand_dims(index, axis=-1), src))


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
    test = MelBandRoformer(dim=64, depth=1, rngs=nnx.Rngs(0))
    output = test(jnp.ones((1, 2, 16000)), deterministic=True)
    print(output.shape)
