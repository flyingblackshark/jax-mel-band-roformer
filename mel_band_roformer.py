from functools import partial
from typing import Any, Sequence, Tuple
import audax.core
import audax.core.stft
from einops import einsum, rearrange, pack, unpack,repeat,reduce
from flax import traverse_util
import jax
import jax.numpy as jnp
import jax.lax as lax
import flax.linen as nn
import audax
import numpy as np
from librosa import filters
def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

class RMSNorm(nn.Module):
  """RMS normalization."""
  dim : int
  dtype: Any = jnp.float32
  weight_dtype: Any = jnp.float32
  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    x = x / jnp.linalg.norm(x,axis=-1,keepdims=True)
    gamma = self.param(
        "gamma",
        nn.initializers.ones,
        (self.dim,),
        self.weight_dtype,
    )

    gamma = jnp.asarray(gamma, self.dtype)
    y = x * gamma * (self.dim ** 0.5)
    return y

class FeedForward(nn.Module):
    dim:int
    mult:int=4
    dropout:float=0.
    @nn.compact
    def __call__(self, x,deterministic):
        dim_inner = int(self.dim * self.mult)
        net = nn.Sequential([
            RMSNorm(self.dim),
            nn.Dense(dim_inner),
            nn.gelu,
            nn.Dropout(self.dropout,deterministic=deterministic),
            nn.Dense(self.dim),
            nn.Dropout(self.dropout,deterministic=deterministic)]
        )
        return net(x)
class Attend(nn.Module):
    dropout:float = 0.
    def setup(self):
        self.attn_dropout = nn.Dropout(self.dropout)

    def __call__(self, q, k, v,deterministic):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        #q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        #scale = default(self.scale, q.shape[-1] ** -0.5)
        scale = q.shape[-1] ** -0.5
        q = q.transpose(0,2,1,3)
        k = k.transpose(0,2,1,3)
        v = v.transpose(0,2,1,3)
        out = nn.dot_product_attention(q,k,v,dropout_rate=0,deterministic=deterministic)

        return out.transpose(0,2,1,3)
        # similarity

        # sim = einsum(q, k,f"b h i d, b h j d -> b h i j") * scale

        # # attention

        # attn = nn.softmax(sim,axis=-1)
        # attn = self.attn_dropout(attn,deterministic=deterministic)

        # # aggregate values

        # out = einsum(attn, v,f"b h i j, b h j d -> b h i d")

        # return out

def get_seq_pos(seq_len, offset = 0):
    return (jnp.arange(seq_len) + offset)# / self.interpolate_factor
def embed_forward(
        t : jnp.ndarray,
        seq_len = None,
        offset = 0,
        dim_head = None,
        freqs = None
    ):
        dim = dim_head
        theta = 10000
        #freqs = freqs = 1. / (theta ** (jnp.arange(0, dim, 2)[:(dim // 2)] / dim))

        freqs = einsum(t, freqs,'..., f -> ... f')
        freqs = repeat(freqs, '... n -> ... (n r)', r = 2)

        return freqs
def rotate_queries_or_keys(t, seq_dim = None, offset = 0, scale = None ,dim_head=None,freqs=None):
    #seq_dim = default(seq_dim, self.default_seq_dim)
    seq_dim = -2
    #assert not self.use_xpos or exists(scale), 'you must use `.rotate_queries_and_keys` method instead and pass in both queries and keys, for length extrapolatable rotary embeddings'
    seq_len = t.shape[seq_dim]

    seq = get_seq_pos(seq_len, offset = offset)

    freqs = embed_forward(seq, seq_len = seq_len, offset = offset , dim_head=dim_head,freqs=freqs)

    if seq_dim == -3:
        freqs = rearrange(freqs, 'n d -> n 1 d')

    return apply_rotary_emb(freqs, t, scale = default(scale, 1.), seq_dim = seq_dim)
def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x1, x2 = jax_unstack(x,axis= -1)
    x = jnp.stack((-x2, x1), axis = -1)
    return rearrange(x, '... d r -> ... (d r)')
def apply_rotary_emb(freqs, t, start_index = 0, scale = 1., seq_dim = -2):

    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    #assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * jnp.cos(freqs) * scale) + (rotate_half(t) * jnp.sin(freqs) * scale)
    out = jnp.concatenate((t_left, t, t_right), axis = -1)

    return out
class Attention(nn.Module):
    dim:int
    heads:int=8
    dim_head:int=64
    dropout:float=0.

    @nn.compact
    def __call__(self, x,deterministic):
        dim_inner = self.heads * self.dim_head
        x = RMSNorm(self.dim)(x)
        temp_x = nn.Dense(dim_inner * 3, use_bias=False,name="to_qkv")(x)
        q, k, v = rearrange(temp_x, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        freqs = self.param(
        "freqs",
        nn.initializers.ones,
        (self.dim_head//2,),
        jnp.float32,
        )

        q = rotate_queries_or_keys(q,dim_head=self.dim_head,freqs=freqs)
        k = rotate_queries_or_keys(k,dim_head=self.dim_head,freqs=freqs)

        out = Attend(dropout=self.dropout,name="attend")(q, k, v,deterministic)

        gates = nn.Dense(self.heads,name="to_gates")(x)
        out = out * nn.sigmoid(rearrange(gates, 'b n h -> b h n 1'))

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = nn.Dense(self.dim, use_bias=False,name="to_out")(out)
        out = nn.Dropout(self.dropout)(out,deterministic=deterministic)
        return out
    
class Transformer(nn.Module):
    dim:int
    depth:int
    dim_head:int=64
    heads:int=8
    attn_dropout:float=0.
    ff_dropout:float=0.
    ff_mult:int=4
    norm_output:bool=True

    def setup(self):
        layers = []

        for _ in range(self.depth):
            attn = Attention(dim=self.dim, dim_head=self.dim_head, heads=self.heads, dropout=self.attn_dropout)

            layers.append([
                attn,
                FeedForward(dim=self.dim, mult=self.ff_mult, dropout=self.ff_dropout)
            ])

        self.norm = RMSNorm(self.dim) 
        self.layers = layers

    def __call__(self, x,deterministic):

        for attn, ff in self.layers:
            x = attn(x,deterministic) + x
            x = ff(x,deterministic) + x

        return self.norm(x)
class BandSplit(nn.Module):
    dim:int
    freqs_per_bands_with_complex:tuple[int]
    freqs_per_bands_with_complex_cum:tuple[int]
    #dim_inputs: Sequence[int]
      
    @nn.compact
    def __call__(self, x):
        to_features = []
        for dim_in in self.freqs_per_bands_with_complex:
          net = nn.Sequential([
              RMSNorm(dim_in),
              nn.Dense(self.dim)
          ])

          to_features.append(net)
        
        x = jnp.split(x,self.freqs_per_bands_with_complex_cum, axis=-1)

        outs = []
        for split_input, to_feature in zip(x, to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return jnp.stack(outs, axis=-2)
def MLP(
    dim_in,
    dim_out,
    dim_hidden=None,
    depth=1,
    activation=nn.tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Dense(layer_dim_out))

        if is_last:
            continue

        net.append(activation)

    return nn.Sequential(net)
def jax_unstack(x, axis=0):
  return [lax.index_in_dim(x, i, axis, keepdims=False) for i in range(x.shape[axis])]
class MaskEstimator(nn.Module):
    dim:int
    dim_inputs: Sequence[int]
    depth:int
    mlp_expansion_factor:int=4
    def setup(self):
        to_freqs = []
        dim_hidden = self.dim * self.mlp_expansion_factor

        for dim_in in self.dim_inputs:
            #net = []
            mlp_layer = MLP(self.dim, dim_in * 2, dim_hidden=dim_hidden, depth=self.depth)
            mlp = nn.Sequential([
                mlp_layer,
                nn.glu]
            )

            to_freqs.append(mlp)
        self.to_freqs = to_freqs

    def __call__(self, x):
        x = jax_unstack(x,axis=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return jnp.concatenate(outs, axis=-1)
# DEFAULT_FREQS_PER_BANDS = [
#     2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#     2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
#     2, 2, 2, 2,
#     4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
#     12, 12, 12, 12, 12, 12, 12, 12,
#     24, 24, 24, 24, 24, 24, 24, 24,
#     48, 48, 48, 48, 48, 48, 48, 48,
#     128, 129,
# ]

# freqs_per_bands_with_complex = []
# for i in range(len(DEFAULT_FREQS_PER_BANDS)):
#     freqs_per_bands_with_complex.append(DEFAULT_FREQS_PER_BANDS[i] * 2 * 2)
# freqs_per_bands_with_complex_cum = np.cumsum(np.asarray(freqs_per_bands_with_complex))

# mel_filter_bank_numpy = filters.mel(sr=44100, n_fft=2048, n_mels=60)
# mel_filter_bank_numpy[0][0] = 1.
# mel_filter_bank_numpy[-1, -1] = 1.
# freqs_per_band = mel_filter_bank_numpy > 0
# freqs = 1025
# repeated_freq_indices = repeat(np.arange(freqs), 'f -> b f', b=60)
# freq_indices = repeated_freq_indices[freqs_per_band]
# #stereo
# freq_indices = repeat(freq_indices, 'f -> f s', s=2)
# freq_indices = freq_indices * 2 + np.arange(2)
# freq_indices = rearrange(freq_indices, 'f s -> (f s)')

# num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
# num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')
# freqs_per_bands_with_complex = tuple(2 * f * 2 for f in num_freqs_per_band.tolist())
# freqs_per_bands_with_complex_cum = np.cumsum(np.asarray(freqs_per_bands_with_complex))

class MelBandRoformer(nn.Module):
    dim:int=384
    depth:int=6
    stereo:bool=True
    num_stems:int=1
    time_transformer_depth:int=1
    freq_transformer_depth:int=1
    linear_transformer_depth:int=0
    num_bands:int=60
    dim_head:int=64
    heads:int=8
    attn_dropout:float=0.1
    ff_dropout:float=0.1
    #flash_attn=True
    dim_freqs_in:int=1025
    stft_n_fft:int=2048
    stft_hop_length:int=441
    # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
    stft_win_length:int=2048
    stft_normalized:int=False
    #stft_window_fn: Optional[Callable] = None,
    mask_estimator_depth:int=3
    multi_stft_resolution_loss_weight:float=1.
    multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256)
    multi_stft_hop_size:int=147
    multi_stft_normalized:bool=False
    # multi_stft_window_fn: Callable = torch.hann_window
    freqs_per_bands_with_complex:tuple[int]=None
    freqs_per_bands_with_complex_cum:tuple[int]=None
    freq_indices:tuple[int]=None
    num_bands_per_freq:tuple[int]=None
    @nn.compact
    def __call__(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False,
            deterministic=False):
        audio_channels = 2 if self.stereo else 1

        if raw_audio.ndim == 2:
          raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        batch, channels, raw_audio_length = raw_audio.shape

        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft


        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = jnp.hanning(self.stft_win_length)

        stft_repr = audax.core.stft.stft(raw_audio, n_fft=self.stft_n_fft,hop_length=self.stft_hop_length,win_length=self.stft_win_length, window=stft_window)
        stft_repr = stft_repr.transpose(0,2,1)
        stft_repr = as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr,'b s f t c -> b (f s) t c')

        batch_arange = jnp.arange(batch)[..., None]
        x = stft_repr[batch_arange, self.freq_indices]

        x = rearrange(x, 'b f t c -> b t (f c)')
        
        x = BandSplit(
            dim=self.dim,
            freqs_per_bands_with_complex=self.freqs_per_bands_with_complex,
            freqs_per_bands_with_complex_cum=self.freqs_per_bands_with_complex_cum
        )(x)

        for i in range(self.depth):
          x = rearrange(x, 'b t f d -> b f t d')
          x, ps = pack([x], '* t d')

          x = Transformer(depth=self.time_transformer_depth,
            name=f"time_transformer_{i}", 
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            norm_output=False)(x,deterministic=deterministic)

          x, = unpack(x, ps, '* t d')
          x = rearrange(x, 'b f t d -> b t f d')
          x, ps = pack([x], '* f d')

          x = Transformer(depth=self.freq_transformer_depth,
            name=f"freq_transformer_{i}",
            dim=self.dim,
            heads=self.heads,
            dim_head=self.dim_head,
            attn_dropout=self.attn_dropout,
            ff_dropout=self.ff_dropout,
            norm_output=False)(x,deterministic=deterministic)

          x, = unpack(x, ps, '* f d')

        out = []
        for _ in range(self.num_stems):
            res = MaskEstimator(
                dim=self.dim,
                dim_inputs=self.freqs_per_bands_with_complex,
                depth=self.mask_estimator_depth
            )(x)
            out.append(res)
        masks = jnp.stack(out,axis=1)
        masks = rearrange(masks, 'b n t (f c) -> b n f t c', c=2)


        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication
        stft_repr = as_complex(stft_repr)
        masks = as_complex(masks)

        scatter_indices = repeat(self.freq_indices, 'f -> b n f t', b=batch, n=self.num_stems, t=stft_repr.shape[-1])
        stft_repr_expanded_stems = repeat(stft_repr, 'b 1 ... -> b n ...', n=self.num_stems,)

        masks_summed = scatter(input=jnp.zeros_like(stft_repr_expanded_stems),dim=2,index=scatter_indices,src=masks,reduce="add")

        denom = repeat(self.num_bands_per_freq, 'f -> (f r) 1', r=channels)

        masks_averaged = masks_summed / jnp.clip(denom,min=1e-8)

        stft_repr = stft_repr * masks_averaged

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=audio_channels)
        t , recon_audio =jax.scipy.signal.istft(stft_repr,nfft=self.stft_n_fft,
            noverlap=self.stft_win_length-self.stft_hop_length,
            nperseg=self.stft_win_length,boundary=False,input_onesided=True)

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=audio_channels, n=self.num_stems)
        if self.num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

def scatter(input, dim, index, src, reduce=None):
    # JAX-port of PyTorch's scatter. See https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html
    # One can simplify this implementation with ndarray.at (ref: https://github.com/google/jax/issues/8487)

    dnums = jax.lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,))
    
    if reduce is None:
        _scatter = jax.lax.scatter
    elif reduce == "add":
        _scatter = jax.lax.scatter_add
    elif reduce == "multiply":
        _scatter = jax.lax.scatter_mul
        
    _scatter = partial(_scatter, dimension_numbers=dnums)
    vmap_inner = partial(jax.vmap, in_axes=(0, 0, 0), out_axes=0)

    for _ in range(len(input.shape)-1):
        _scatter = vmap_inner(_scatter)
    swap = lambda x: jnp.swapaxes(x, dim, -1)
    input, index, src = list(map(swap, (input, index, src)))
    return swap(_scatter(input, jnp.expand_dims(index, axis=-1), src))

def as_complex(x):
   assert x.shape[-1] == 2
   return jax.lax.complex(x[...,0], x[...,1])
def as_real(x):
    if not jnp.issubdtype(x.dtype, jnp.complexfloating):
        return x

    xr = jnp.zeros(x.shape+(2,), dtype=x.real.dtype)
    xr = xr.at[...,0].set(x.real)
    xr = xr.at[...,1].set(x.imag)
    return xr

if __name__ =="__main__":
    test = MelBandRoformer()
    # init_arr = jnp.ones((1,2,16000))
    # rngs = {'params': jax.random.key(0), 'other_rng': jax.random.key(1)}
    # params_init = test.init(rngs,init_arr)["params"]
    # flatten_param = traverse_util.flatten_dict(params_init,sep='.')

    from convert import load_params
    params = load_params()
    output = test.apply({"params":params},jnp.ones((1,2,16000)),deterministic=True)
    flatten_param = traverse_util.flatten_dict(params,sep='.')
    print(output)
    #print(output)