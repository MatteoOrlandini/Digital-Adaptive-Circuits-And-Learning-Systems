import os
import numpy
import torch
from preprocessing import * 
from mel_spectrogram import * 
from tqdm import tqdm

def find_classes(reader, C, K, Q = 16):
    classes = []

    # taking at most C words from all training readers ones
    if (C <= len(reader['words'])):
        reader_words = random.sample(reader['words'], C)  
    else:
        reader_words = random.sample(reader['words'], len(reader['words']))  

    for word in reader_words:
        # numpy.arange returns evenly spaced values within a given interval.
        # create an array of index to get the start, end and folder of the same index
        index_array = list(numpy.arange(len(word['start'])))

        # random sample only K + Q indexes
        index_array = random.sample(index_array, K + Q)

        instance_start = []
        instance_end = []
        instance_folder = []
        
        # sample K instances from every C word class
        for index in index_array:
            # get the start, end and folder of the same index
            instance_start.append(word['start'][index])
            instance_end.append(word['end'][index])
            instance_folder.append(word['folders'][index])

        # append the new word of K + Q instances
        classes.append( {'word'   : word['word'], \
                        'start'   : instance_start,\
                        'end'     : instance_end, \
                        'folders' : instance_folder})

    #write_json_file("Classi/training_words_of_"+ reader['reader_name'] +".json", classes)
    return classes

def save_dataset(readers, C, K, Q, folder_name, dataset_path, audio_file_name):
    for reader in tqdm(readers, position = 0):
        if not (os.path.exists(folder_name + reader['reader_name'])):
            try:
                os.mkdir(folder_name + reader['reader_name'])
                classes = find_classes(reader, C, K, Q)
                for item in tqdm(classes, position = 1, leave = False):
                    spectrograms = np.empty([0, 128, 51])
                    for i in range(len(item['start'])):
                        start_in_sec = item['start'][i]/1000 # conversion from milliseconds to seconds
                        end_in_sec = item['end'][i]/1000     # conversion from milliseconds to seconds
                        word_center_time = (start_in_sec + end_in_sec)/2
                        audio_file_path = dataset_path + item['folders'][i] + "/" + audio_file_name
                        if (os.path.exists(audio_file_path)):
                            item_spectrogram = compute_melspectrogram(audio_file_path, word_center_time)
                            spectrograms = np.concatenate((spectrograms, [item_spectrogram]), axis = 0)
                    if (spectrograms.shape[0] == K + Q):
                        torch_tensor = torch.FloatTensor(spectrograms)
                        torch.save(torch_tensor, folder_name + reader['reader_name'] + "/" + item['word'] + ".pt")
            except OSError as error:
                print(error)   

def batch_sample(feature_folder, C, K, Q):
    support =  torch.empty([0 ,K, 128, 51])
    query = torch.empty([0, Q, 128, 51])
    reader_path = []
    for entry in os.scandir(feature_folder):
        reader_path.append(entry.path)
        #print(entry.path)
    words = []
    while (len(words) < C):
        folder = random.sample(reader_path, 1)
        #print("folder:",folder[0])
        words = []
        for word in os.scandir(folder[0]):
            words.append(word.path)
    words = random.sample(words, C)
    #print("words:",words)
    for word in words:
        spectrogram = torch.load(word)
        x_dim, y_dim, z_dim = spectrogram.shape
        instances_number = (spectrogram.shape)[0]
        index = random.sample(list(torch.arange(instances_number)), K + Q)
        spectrogram_buf = torch.empty([0, 128, 51])
        for i in index:
            spectrogram_buf = torch.cat((spectrogram_buf, (spectrogram[i, :, :]).view(1, y_dim, z_dim)), axis = 0)
        support =  torch.cat((support, (spectrogram_buf[:K]).view(1, K, y_dim, z_dim)), axis = 0)
        query = torch.cat((query, (spectrogram_buf[K:K+Q]).view(1, Q, y_dim, z_dim)), axis = 0)
    return query, support


if __name__ == "__main__":
    C = 10 # classes
    K = 10 # instances per class
    Q = 16

    training_feature_folder_name = "Training_features/"
    validation_feature_folder_name = "Validation_features/"
    dataset_path = "./Dataset/English spoken wikipedia/english/"
    audio_file_name = "audio.ogg"

    training_readers = read_json_file("training_readers.json")
    validation_readers = read_json_file("validation_readers.json")

    if not (os.path.exists(training_feature_folder_name)):
        try:
            os.mkdir(training_feature_folder_name)
        except OSError as error:
            print(error)   

    #save_dataset(training_readers, C, K, Q, training_feature_folder_name, dataset_path, audio_file_name)
    
    if not (os.path.exists(validation_feature_folder_name)):
        try:
            os.mkdir(validation_feature_folder_name)
        except OSError as error:
            print(error) 

    #save_dataset(validation_readers, C, K, Q, validation_feature_folder_name, dataset_path, audio_file_name)