from json_manager import *
from mel_spectrogram import *
import random
import numpy
import torch
import os

test_readers= read_json_file('test_readers.json')
reader_paths = read_json_file('reader_paths.json')
readers_list=[]

for reader in test_readers:
    readers_list.append(reader['reader_name'])

for reader in readers_list:
    for item in reader_paths:
        if reader==item['reader_name']:
            audios = item['folder']
    for item in audios:
        path = "/Dataset/English spoken wikipedia/english/" + item + "/target_words.json"
        words=read_json_file(path)
        for word in words:
            spectrograms = np.empty([0, 128, 51])
            if len(word['start'])>16:
                instances=random.sample(numpy.arange(len(word['start'])),16)
            else:
                instances =random.sample(numpy.arange(len(word['start'])))
                for i in instances:
                    start_in_sec = word['start'][i]/1000 # conversion from milliseconds to seconds
                    end_in_sec = word['end'][i]/1000     # conversion from milliseconds to seconds
                    # calculation of the center time of the word
                    word_center_time = (start_in_sec + end_in_sec)/2
                    # path of the audio file
                    audio_file_path = "/Dataset/English spoken wikipedia/english/" + item + "/audio.ogg"
                    if (os.path.exists(audio_file_path)):
                        # compute the 128 bit log mel-spectrogram
                        item_spectrogram = compute_melspectrogram(audio_file_path, word_center_time)
                        # construction of a spectrogram tensor
                        spectrograms = np.concatenate((spectrograms, [item_spectrogram]), axis = 0)
                # save the spectrograms tensor only if the first dimension is higher than K + Q, 
                # that is when the word has at least K + Q instances
                #if (spectrograms.shape[0] >= K + Q):
                    # conversion from numpy array to torch.FloatTensor
                    torch_tensor = torch.FloatTensor(spectrograms)
                    # save the torch tensor
                    torch.save(torch_tensor, "Test_features" + reader + "/" + word['word'] + ".pt")
