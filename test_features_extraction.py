from json_manager import *
from mel_spectrogram import *
import random
import numpy
import torch
import os
from tqdm import tqdm

test_readers = read_json_file('test_readers.json')
reader_paths = read_json_file('readers_paths.json')
readers_list = []
audio_folders = []
test_feature_folder_name = "Test_features/"
dataset_path = "./Dataset/English spoken wikipedia/english/"

for reader in test_readers:
    readers_list.append(reader['reader_name'])

for reader_name in tqdm(readers_list, position = 0, desc = "test readers"):
    for item in reader_paths:
        if reader_name == item['reader_name']:
            audio_folders += item['folder']

for audio_folder in tqdm(audio_folders, position = 0, desc = "audio folders"):
    if not (os.path.exists(test_feature_folder_name + audio_folder)):
        try:
            os.mkdir(test_feature_folder_name + audio_folder)
            path = dataset_path + audio_folder + "/target_words.json"
            words = read_json_file(path)
            for word in words:
                spectrograms = np.empty([0, 128, 51])
                if (len(word['start']) > 16):
                    indices = random.sample(list(numpy.arange(len(word['start']))), 16)
                else:
                    indices = random.sample(list(numpy.arange(len(word['start']))), len(word['start']))
                for i in indices:
                    start_in_sec = word['start'][i]/1000 # conversion from milliseconds to seconds
                    end_in_sec = word['end'][i]/1000     # conversion from milliseconds to seconds
                    # calculation of the center time of the word
                    word_center_time = (start_in_sec + end_in_sec)/2
                    # path of the audio file
                    audio_file_path = dataset_path + audio_folder + "/audio.ogg"
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
                torch.save(torch_tensor, test_feature_folder_name + audio_folder + "/" + word['word'] + ".pt")
            
        except OSError as error:
            print(error)   
