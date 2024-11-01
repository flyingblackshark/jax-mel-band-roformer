import argparse
from mel_band_roformer import MelBandRoformer
from convert import load_params
import librosa
import numpy as np
import jax.numpy as jnp
import soundfile as sf
import glob
import os
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils,multihost_utils
from functools import partial
import jax
import time
from librosa import filters
import numpy as np
from einops import einsum, rearrange, pack, unpack,repeat,reduce
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

def run_folder(args):
    start_time = time.time()
    freq_indices,num_bands_per_freq,freqs_per_bands_with_complex,freqs_per_bands_with_complex_cum = pre_compute()
    model = MelBandRoformer(precision=jax.lax.Precision.DEFAULT,
                            freq_indices=freq_indices,
                            num_bands_per_freq=num_bands_per_freq,
                            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
                            freqs_per_bands_with_complex_cum=freqs_per_bands_with_complex_cum)
    params = load_params(args.start_check_point)
    
    all_mixtures_path = glob.glob(args.input_folder + '/*.*')
    all_mixtures_path.sort()
    num_audios = len(all_mixtures_path)
    print('Total files found: {}'.format(num_audios))

    if not os.path.isdir(args.store_dir):
        os.mkdir(args.store_dir)

    device_mesh = mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices=device_mesh, axis_names=('data'))
    i = 0

    for path in all_mixtures_path:
        mix, sr = librosa.load(path, sr=44100, mono=False)
        if len(mix.shape) == 1:
            mix = np.stack([mix, mix], axis=0)
        file_name, _ = os.path.splitext(os.path.basename(path))
        print(f"reading: {file_name}")

        res = demix_track(model,params,mix,mesh)
        res = np.asarray(res)
        estimates = res.squeeze(0)
        i+=1
        estimates_now = estimates.transpose(1,0)
        estimates_now = estimates_now / np.max(estimates_now)
        if jax.process_index() == 0:
            output_file = os.path.join(args.store_dir, f"{file_name}_vocal.wav")
            sf.write(output_file, estimates_now, sr, subtype = 'FLOAT')
            print(f"{i} write {output_file}")

    print("Elapsed time: {:.2f} sec".format(time.time() - start_time))



def demix_track(model, params,mix,mesh):
    #default chunk size 
    C = 352768  #config.audio.chunk_size
    N = 4 #config.inference.num_overlap
    fade_size = C // 10
    step = int(C // N)
    border = C - step
    batch_size_per_device = 8 #config.inference.batch_size
    device_count = jax.device_count()
    batch_size = batch_size_per_device * device_count

    x_sharding = NamedSharding(mesh,PartitionSpec('data'))
    @partial(jax.jit, in_shardings=(None,x_sharding),
                    out_shardings=x_sharding)
    def model_apply(params, x):
        return model.apply({'params': params}, x , deterministic=True)
    length_init = mix.shape[-1]

    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix =np.pad(mix, ((0,0),(border, border)), mode='reflect')
    def _getWindowingArray(window_size, fade_size):
        fadein = np.linspace(0, 1, fade_size)
        fadeout = np.linspace(1, 0, fade_size)
        window = np.ones(window_size)
        window[-fade_size:] = (window[-fade_size:]*fadeout)
        window[:fade_size] = (window[:fade_size]*fadein)
        return window
    # windowingArray crossfades at segment boundaries to mitigate clicking artifacts
    windowingArray = _getWindowingArray(C, fade_size)

    req_shape = (1, ) + tuple(mix.shape)

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
                part = np.pad(part,((0,0),(0,C-length)),mode='reflect')
            else:
                part = np.pad(part,((0,0),(0,C-length)),mode='constant')
        batch_data.append(part)
        batch_locations.append((i, length))
        i += step

        if len(batch_data) >= batch_size or (i >= mix.shape[1]):
            arr = np.stack(batch_data, axis=0)
            B_padding = max((batch_size-len(batch_data)),0)
            arr = np.pad(arr,((0,B_padding),(0,0),(0,0)))
            arr = multihost_utils.host_local_array_to_global_array(
                arr, mesh, x_sharding
            )

            # infer
            with mesh:
                x = model_apply(params,arr)
            x = x[:batch_size-B_padding]
            window = windowingArray
            if i - step == 0:  # First audio chunk, no fadein
                window[:fade_size] = 1
            elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                window[-fade_size:] = 1

            for j in range(len(batch_locations)):
                start, l = batch_locations[j]
                result[..., start:start+l] += (x[j][..., :l] * window[..., :l])
                counter[..., start:start+l] += (window[..., :l])

            batch_data = []
            batch_locations = []

    estimated_sources = result / counter
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    #estimated_sources = jnp.where(jnp.isnan(estimated_sources), 0, estimated_sources)

    if length_init > 2 * border and (border > 0):
        # Remove pad
        estimated_sources = estimated_sources[..., border:-border]
    return estimated_sources

if __name__ == "__main__":
    jax.distributed.initialize()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_check_point", type=str, default='MelBandRoformer_vocal.ckpt', help="Initial checkpoint to valid weights")
    parser.add_argument("--input_folder",default="./input", type=str, help="folder with mixtures to process")
    parser.add_argument("--store_dir", default="./output", type=str, help="path to store results as wav file")
    parser.add_argument("--disable_detailed_pbar", action='store_true', help="disable detailed progress bar")
    args = parser.parse_args()
    run_folder(args)