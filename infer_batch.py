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
import jaxloudnorm as jln
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils
from functools import partial
import jax
from jax.experimental.compilation_cache import compilation_cache as cc
import time
cc.set_cache_dir("./jax_cache")

def run_folder(args,verbose=False):
    start_time = time.time()
    model = MelBandRoformer(precision=jax.lax.Precision.DEFAULT)
    params = load_params(args.start_check_point)
    model = (model,params)
    
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    all_mixtures_path.sort()
    num_audios = len(all_mixtures_path)
    print('Total files found: {}'.format(num_audios))

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    if not verbose:
        all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

    # if args.disable_detailed_pbar:
    #     detailed_pbar = False
    # else:
    #     detailed_pbar = True
    device_mesh = mesh_utils.create_device_mesh((4,))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    mixtures = iter(all_mixtures_path)
    i = 0

    
    
    while i < num_audios:
        bigmix = None
        file_name_arr = []
        length_arr = [0]
        while i < num_audios:
            path = next(mixtures)
            mix, sr = librosa.load(path, sr=44100, mono=False)
            if len(mix.shape) == 1:
                mix = jnp.stack([mix, mix], axis=0)
            length_arr.append(mix.shape[1])
            file_name, _ = os.path.splitext(os.path.basename(path))
            file_name_arr.append(file_name)
            print(f"reading: {file_name}")
            if bigmix is None:
                bigmix = mix
            else:
                bigmix = jnp.concatenate([bigmix,mix],axis=1)
            print(f"bigmix length now: {bigmix.shape[1]}")
            i+=1

            if bigmix.shape[1] >= 352768 * 64:
                break

        res = demix_track(model,bigmix,mesh, pbar=False)
        estimates = res.squeeze(0)
        length_arr = np.asarray(length_arr)
        length_arr = np.cumsum(length_arr)
        for j in range(len(file_name_arr)):
            estimates_now = estimates.transpose(1,0)
            estimates_now = estimates_now[length_arr[j]:length_arr[j+1]]
            estimates_now = estimates_now / np.max(estimates_now)
            output_file = os.path.join(args.store_dir, f"{file_name_arr[j]}_vocal.wav")
            sf.write(output_file, estimates_now, sr, subtype = 'FLOAT')
            print(f"{i}/{num_audios} write {output_file}")

    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))
def _getWindowingArray(window_size, fade_size):
    fadein = jnp.linspace(0, 1, fade_size)
    fadeout = jnp.linspace(1, 0, fade_size)
    window = jnp.ones(window_size)
    window = window.at[-fade_size:].set(window[-fade_size:]*fadeout)
    window = window.at[:fade_size].set(window[:fade_size]*fadein)
    return window

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

    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

   
    # if config.training.target_instrument is not None:
    req_shape = (1, ) + tuple(mix.shape)
    # else:
    #     req_shape = (len(config.training.instruments),) + tuple(mix.shape)

    result = jnp.zeros(req_shape, dtype=jnp.float32)
    counter = jnp.zeros(req_shape, dtype=jnp.float32)
    i = 0
    batch_data = []
    batch_locations = []
    progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None
    # meter = jln.Meter(44100) # create BS.1770 meter
    # measure_loudness_jit = jax.jit(jax.vmap(meter.integrated_loudness)
    #                                 ,in_shardings=(x_sharding)
    #                                 ,out_shardings=(x_sharding))
    # normalize_loudness_jit = jax.jit(jax.vmap(jln.normalize.loudness, in_axes=(0, 0, 0))
    #                                 ,in_shardings=(x_sharding,x_sharding,x_sharding)
    #                                 ,out_shardings=(x_sharding))

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
            #pre record loudness
            
            #loudness_old = measure_loudness_jit(arr.transpose(0,2,1))
            # for safety
            # if loudness_old > -16:
            #     loudness_old -= 2
            # infer
            with mesh:
                x = model_apply(params,arr)
            #restore loudness
            # loudness_new = measure_loudness_jit(x.transpose(0,2,1))
            # x = normalize_loudness_jit(x.transpose(0,2,1), loudness_new, loudness_old).transpose(0,2,1)
            
            x = x[:batch_size-B_padding]
            window = windowingArray
            if i - step == 0:  # First audio chunk, no fadein
                window = window.at[:fade_size].set(1)
            elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                window = window.at[-fade_size:].set(1)

            for j in range(len(batch_locations)):
                start, l = batch_locations[j]
                result = result.at[..., start:start+l].set(result[..., start:start+l] + x[j][..., :l] * window[..., :l])
                counter = counter.at[..., start:start+l].set(counter[..., start:start+l] + window[..., :l])

            batch_data = []
            batch_locations = []

        if progress_bar:
            progress_bar.update(step)

    if progress_bar:
        progress_bar.close()

    estimated_sources = result / counter
    estimated_sources = jnp.nan_to_num(estimated_sources,copy=False,nan=0)

    if length_init > 2 * border and (border > 0):
        # Remove pad
        estimated_sources = estimated_sources[..., border:-border]
    return estimated_sources

def proc_folder(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_check_point", type=str, default='MelBandRoformer_vocal.ckpt', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder",default="./input", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default="./output", type=str, help="path to store results as wav file")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    args = parser.parse_args()
    run_folder(args,verbose=True)


if __name__ == "__main__":
    jax.distributed.initialize()
    proc_folder(None)

