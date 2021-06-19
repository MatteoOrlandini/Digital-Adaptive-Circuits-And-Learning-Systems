import librosa
import librosa.display
import soundfile
import audioread
import numpy as np
import matplotlib.pyplot as plt

def compute_melspectrogram(filename, word_center):
    # get the original sampling rate for a given file
    #original_sample_rate = librosa.get_samplerate(filename)
    #print('Original sample rate:', original_sample_rate)
    # audio recordings are downsampled to a sampling rate of 16 kHz.
    sample_rate = 16000
    #librosa.load loads the audio file as a floating point time series. y: audio time series, sr: sample rate

    y, sr = librosa.load(path = filename, sr = sample_rate, offset = word_center - 0.25, duration = 0.5)

    soundfile.write('stone.wav', y, sr)
    #soundfile.write('prova.wav', y[int(word_center * sr - 0.25 * sr): int(word_center * sr + 0.25 * sr)], sr)
    
    #print('Data size:', y.size)
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

if __name__ == "__main__":
    source_path = "./Dataset/English spoken Wikipedia/english/"
    folder = "(I_Can%27t_Get_No)_Satisfaction/"
    file_name = "audio.ogg"

    # word : stone
    start = 50350   # in milliseconds
    end = 50780     # in milliseconds

    start_in_seconds = start/1000    # in seconds
    end_in_seconds = end/1000        # in seconds
    word_center = (start_in_seconds + end_in_seconds)/2    # in seconds

    compute_melspectrogram(source_path + folder + file_name, word_center)