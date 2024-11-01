from librosa import filters
import numpy as np
from einops import einsum, rearrange, pack, unpack,repeat,reduce
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
breakpoint()