import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

filename = "./Dataset/English spoken Wikipedia with audio/english/Alice_Ingley/audio.ogg"
# audio recordings are downsampled to a sampling rate of 16 kHz.
sample_rate = 16000
# librosa.load loads the audio file as a floating point time series. y: audio time series, sr: sample rate
y, sr = librosa.load(path = filename, sr = sample_rate) 
print('Sample rate:', sr)
print('Data size:', y.size)
# We use a window length of 25 ms, hop size of 10 ms and a fast Fourier transform size of 64 ms.
window_length = int(0.025*sample_rate)
hop_size = int(0.01*sample_rate)
fft_size = int(0.064*sample_rate)
# For each word instance, we take a half-second context window centered on the word and compute a 128 bin log-mel-spectrogram
mels_number = 128
fig, ax = plt.subplots()
# librosa.feature.melspectrogram computes a mel-scaled spectrogram.
mel_spectrogram = librosa.feature.melspectrogram(y, sr = sample_rate, n_fft = fft_size, hop_length = hop_size, win_length = window_length, n_mels = mels_number)
# librosa.power_to_db converts a power spectrogram to a dB-scale spectrogram.
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
# display the log_mel_spectrogram
img = librosa.display.specshow(log_mel_spectrogram, x_axis = "time", y_axis = "mel", sr = sample_rate, ax = ax)
ax.set(title = 'Mel spectrogram display')
fig.colorbar(img, ax = ax)
plt.show()