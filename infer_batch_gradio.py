import argparse
import flax
from mel_band_roformer import MelBandRoformer
from convert import load_params
import librosa
import numpy as np
import jax.numpy as jnp
import soundfile as sf
import glob
import os
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
from functools import partial
import jax
from librosa import filters
import numpy as np
from jax.experimental.compilation_cache import compilation_cache as cc
from einops import einsum, rearrange, pack, unpack,repeat,reduce
import time
cc.set_cache_dir("./jax_cache")
def pre_compute():
    mel_filter_bank_numpy = filters.mel(sr=44100, n_fft=2048, n_mels=60)
    mel_filter_bank_numpy[0][0] = 1.
    mel_filter_bank_numpy[-1, -1] = 1.
    freqs_per_band = mel_filter_bank_numpy > 0
    freqs = 1025
    repeated_freq_indices = repeat(np.arange(freqs), 'f -> b f', b=60)
    freq_indices = repeated_freq_indices[freqs_per_band]
    #stereo
    freq_indices = repeat(freq_indices, 'f -> f s', s=2)
    freq_indices = freq_indices * 2 + np.arange(2)
    freq_indices = rearrange(freq_indices, 'f s -> (f s)')

    num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
    num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')
    freqs_per_bands_with_complex = tuple(2 * f * 2 for f in num_freqs_per_band.tolist())
    freqs_per_bands_with_complex_cum = np.cumsum(np.asarray(freqs_per_bands_with_complex))
    return freq_indices,num_bands_per_freq,freqs_per_bands_with_complex,freqs_per_bands_with_complex_cum

def run_folder(input_audio):
    freq_indices,num_bands_per_freq,freqs_per_bands_with_complex,freqs_per_bands_with_complex_cum = pre_compute()
    model = MelBandRoformer(freq_indices=freq_indices,
                            num_bands_per_freq=num_bands_per_freq,
                            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
                            freqs_per_bands_with_complex_cum=freqs_per_bands_with_complex_cum)
    params = load_params("/root/jax-mel-band-roformer/MelBandRoformer.ckpt")
    model = (model,params)
    device_mesh = mesh_utils.create_device_mesh((4,))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    mix, sr = librosa.load(input_audio, sr=44100, mono=False)
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)

    res = demix_track(model,mix,mesh, pbar=False)
    res = np.asarray(res)
    estimates = res.squeeze(0)
    estimates_now = estimates.transpose(1,0)
    estimates_now = estimates_now / np.max(estimates_now)
    #estimates_now = np.split(estimates_now,axis=-1)
    #output = [(audio,44100) for audio in estimates]
    return 44100,estimates_now



def demix_track(model, mix,mesh, pbar=False):
    model , params = model
    #default chunk size 
    C = 352768  #config.audio.chunk_size
    N = 4 #config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size = 32 #config.inference.batch_size

    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    @partial(jax.jit, in_shardings=(None,x_sharding),
                    out_shardings=x_sharding)
    def model_apply(params, x):
        return model.apply({'params': params}, x , deterministic=True)
    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix =jnp.pad(mix, ((0,0),(border, border)), mode='reflect')
    def _getWindowingArray(window_size, fade_size):
        fadein = np.linspace(0, 1, fade_size)
        fadeout = np.linspace(1, 0, fade_size)
        window = np.ones(window_size)
        window[-fade_size:] = (window[-fade_size:]*fadeout)
        window[:fade_size] = (window[:fade_size]*fadein)
        return window
    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

   
    # if config.training.target_instrument is not None:
    req_shape = (1, ) + tuple(mix.shape)
    # else:
    #     req_shape = (len(config.training.instruments),) + tuple(mix.shape)

    result = np.zeros(req_shape, dtype=jnp.float32)
    counter = np.zeros(req_shape, dtype=jnp.float32)
    i = 0
    batch_data = []
    batch_locations = []

    while i < mix.shape[1]:
        part = mix[:, i:i + C]
        length = part.shape[-1]
        if length < C:
            if length > C // 2 + 1:
                part = jnp.pad(part,((0,0),(0,C-length)),mode='reflect')
            else:
                part = jnp.pad(part,((0,0),(0,C-length)),mode='constant')
        batch_data.append(part)
        batch_locations.append((i, length))
        i += step

        if len(batch_data) >= batch_size or (i >= mix.shape[1]):
            arr = jnp.stack(batch_data, axis=0)
            B_padding = max((batch_size-len(batch_data)),0)
            arr = jnp.pad(arr,((0,B_padding),(0,0),(0,0)))
            # infer
            with mesh:
                x = model_apply(params,arr)
            window = windowingArray
            if i - step == 0:  # First audio chunk, no fadein
                window[:fade_size] =1
            elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                window[-fade_size:] =1
            
            total_add_value = jax.jit(jnp.multiply)(x[..., :C],window)
            total_add_value = total_add_value[:batch_size-B_padding]
            total_add_value = np.asarray(total_add_value)
            for j in range(len(batch_locations)):
                start, l = batch_locations[j]
                result[..., start:start+l] += total_add_value[j][..., :l]
                counter[..., start:start+l]+= window[..., :l]

            batch_data = []
            batch_locations = []

    estimated_sources = result / counter
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    if length_init > 2 * border and (border > 0):
        # Remove pad
        estimated_sources = estimated_sources[..., border:-border]
    return estimated_sources


import gradio as gr
if __name__ == "__main__":
    jax.distributed.initialize()
    # 创建Gradio界面
    iface = gr.Interface(
        fn=run_folder,                       # 处理函数
        inputs=gr.Audio(type="filepath"),  # 输入类型为音频文件上传
        outputs=gr.Audio(type="numpy"),             # 输出类型为音频文件
        title="人声提取",                          # 界面标题
        description="上传音频文件，输出提取后的人声" # 描述
    )

    # 启动界面
    iface.launch()
