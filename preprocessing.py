import random
import os
from json_manager import *
import numpy
import torch
from preprocessing import * 
from mel_spectrogram import * 
from tqdm import tqdm

def find_valid_readers(min_classes = 2, min_instances_per_class = 26):
    """
    find_valid_readers returns a list of dict with 'word', 'start', 'end' and 'folders' keys
    for each reader name only for the readers with at least min_classes words and min_instances_per_class
    instances per word

    Parameters:
    min_classes (int) (default 2): minimum number of classes per each reader
    min_instances_per_class (int) (default 26): minimum number of instances per class

    Returns:
    valid_readers (list of dict): it contains only the readers with at least min_classes words and 
    min_instances_per_class instances per word
    """
    # read the json with the words of each reader
    readers = read_json_file("readers_words.json")

    valid_readers = []

    # for each reader find if it's a valid reader only if he reads C words for K + Q instances
    for reader in readers:    
        valid_words = []
        is_valid_reader = False
        number_of_valid_words = 0

        # for each word of a reader find if there are for K + Q instances
        for word in reader['words']:
            start = []
            end = []
            folders = []

            for item in word['folders']:
                start += item['start']
                end += item['end']
                folders += [item['folder']]*len(item['start'])

            new_word = {'word'   : word['word'], \
                        'start'  : start, \
                        'end'    : end, \
                        'folders': folders}

            # if there are at least 26 (default) instances append a new valid word
            if (len(new_word['start']) >= min_instances_per_class):
                number_of_valid_words += 1
                valid_words.append(new_word)

        # if the number of valid words is at least min_classes, then the reader is valid
        if (number_of_valid_words >= min_classes):
            is_valid_reader = True

        # if the reader is valid then add the reader to the valid readers list
        if (is_valid_reader):
            valid_readers.append({'reader_name' : reader['reader_name'],\
                                    'words': valid_words})

    #write_json_file("valid_readers.json", valid_readers)
    return valid_readers

def create_training_validation_test_readers(valid_readers, number_of_training_readers, number_of_test_readers, number_of_validation_readers):
    """
    create_training_validation_test_readers split the valid readers into training+validation readers and
    test readers

    Parameters:
    valid_readers (list of dict): list of the valid readers
    number_of_training_readers (int): size of the training readers
    number_of_validation_readers (int): size of the validation readers
    number_of_test_readers (int): size of the test readers

    Returns:
    training_validation_readers (list of dict): it contains the training and validation readers
    splitted from valid_readers
    test_readers (list of dict): it contains the test readers splitted from valid_readers
    """

    # take the first ("number_of_training_readers" + number_of_validation_readers") elements
    # of "valid_readers" to create the training and validation readers
    training_validation_readers = valid_readers[0 : number_of_training_readers + number_of_validation_readers]

    # take the last "number_of_test_readers" of "valid_readers" to create the test readers
    test_readers = valid_readers[number_of_training_readers + number_of_validation_readers:]

    #write_json_file("training_validation_readers.json", training_readers)
    #write_json_file("test_readers.json", test_readers)

    return training_validation_readers, test_readers

def find_classes(reader, max_class_number, max_instances_number):
    """
    find_classes returns classes, a list of dict that has 'word', 'start', 'end', 'folders' keys.
    'word' value is a string, the word name.
    'start', 'end', 'folders' values are lists. The i-th element of 'start' and 'end' value
    corresponds to the i-th element of 'folder' value.
    If the reader contains less words than max_class_number then random sample the words, 
    otherwise random sample max_class_number words.

    Parameters:
    reader (string): one reader from the json file of readers 
    max_class_number (int): maximum class size
    max_instances_number (int): maximum instance size

    Returns:
    classes (list): 
    """
    classes = []

    # taking at most C words from the reader
    # If the reader contains more words than C, random sample C words
    if (len(reader['words']) >= max_class_number):
        reader_words = random.sample(reader['words'], max_class_number)
    # If the reader contains less words than C, random sample the words
    else:
        reader_words = random.sample(reader['words'], len(reader['words']))  

    for word in reader_words:
        # numpy.arange returns evenly spaced values within a given interval.
        # create an array of index to get the start, end and folder of the same index
        index_array = list(numpy.arange(len(word['start'])))
        
        # taking at most max_instances_number instances from each word
        # If the word contains more instances than max_instances_number, random sample max_instances_number instances
        if (len(index_array) >= max_instances_number):
            index_array = random.sample(index_array, max_instances_number)
        # If the word contains less instances than max_instances_number, random sample the index_array
        else:
            index_array = random.sample(index_array, len(index_array))  

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
        classes.append({'word'    : word['word'], \
                        'start'   : instance_start,\
                        'end'     : instance_end, \
                        'folders' : instance_folder})

    #write_json_file("Classi/training_words_of_"+ reader['reader_name'] +".json", classes)
    return classes

def save_dataset(readers, folder_name, dataset_path, audio_file_name, max_class_number, max_instances_number):
    """
    save_dataset saves a pytorch tensor for each word of a reader in a folder named as the 
    reader name.

    Parameters:
    readers (list of dict): the list of the training and validation readers saved in training_validation_readers.json
    folder_name (string): name of the folder in which save the features
    dataset_path (string): path of the Spoken Wikipedia Corpora dataset
    audio_file_name (string): name of the ".ogg" audio file 
    max_class_number (int): maximum class size
    max_instances_number (int): maximum instance size

    Returns:
    """
    for reader in tqdm(readers, position = 0):
        if not (os.path.exists(folder_name + reader['reader_name'])):
            try:
                os.mkdir(folder_name + reader['reader_name'])
                classes = find_classes(reader, max_class_number, max_instances_number)
                for item in tqdm(classes, position = 1, leave = False):
                    spectrograms = np.empty([0, 128, 51])
                    for i in range(len(item['start'])):
                        start_in_sec = item['start'][i]/1000 # conversion from milliseconds to seconds
                        end_in_sec = item['end'][i]/1000     # conversion from milliseconds to seconds
                        # calculation of the center time of the word
                        word_center_time = (start_in_sec + end_in_sec)/2
                        # path of the audio file
                        audio_file_path = dataset_path + item['folders'][i] + "/" + audio_file_name
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
                        torch.save(torch_tensor, folder_name + reader['reader_name'] + "/" + item['word'] + ".pt")
            except OSError as error:
                print(error)   

def save_test_dataset(test_readers, reader_paths, test_feature_folder_name, dataset_path, audio_file_name):
    """
    save_test_dataset saves a pytorch tensor for each word of a test reader in a folder named 
    as the audio of the test reader.

    Parameters:
    test_readers (list of dict): the list of the test readers saved in test_readers.json
    test_feature_folder_name (string): name of the folder in which save the features
    dataset_path (string): path of the Spoken Wikipedia Corpora dataset
    audio_file_name (string): name of the ".ogg" audio file 

    Returns:
    """
    readers_list = []
    audio_folders = []

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
                        audio_file_path = dataset_path + audio_folder + audio_file_name
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


if __name__ == "__main__":
    """
    min_classes = 2 # minimum number of classes
    min_instances_per_class = 26 # minimum number of instances per class
    
    valid_readers = find_valid_readers(min_classes, min_instances_per_class)
    # random sample the valid readers
    valid_readers = random.sample(valid_readers, len(valid_readers))

    # The readers are partitioned into training, validation, and test sets with a 138:15:30 ratio
    number_of_training_readers = int(138/183*len(valid_readers))
    number_of_test_readers = int(30/183*len(valid_readers))
    number_of_validation_readers = int(15/183*len(valid_readers))

    # The valid readers are partitioned into training, validation, and test readers
    training_validation_readers, test_readers = create_training_validation_test_readers(valid_readers, \
                                                                                                number_of_training_readers, \
                                                                                                number_of_test_readers, \
                                                                                                number_of_validation_readers)
    """
    # create the folder for the training and validation features
    training_validation_feature_folder_name = "Training_validation_features/"
    
    if not (os.path.exists(training_validation_feature_folder_name)):
        try:
            os.mkdir(training_validation_feature_folder_name)
        except OSError as error:
            print(error)   

    max_class_number = 32
    max_instances_number = 64
    
    dataset_path = "./Dataset/English spoken wikipedia/english/"
    audio_file_name = "audio.ogg"

    training_validation_readers = read_json_file("training_validation_readers.json")

    save_dataset(training_validation_readers, training_validation_feature_folder_name, dataset_path, audio_file_name, max_class_number, max_instances_number)

    test_readers = read_json_file('test_readers.json')
    reader_paths = read_json_file('readers_paths.json')

    # create the folder for the test features
    test_feature_folder_name = "Test_features/"

    if not (os.path.exists(test_feature_folder_name)):
        try:
            os.mkdir(test_feature_folder_name)
        except OSError as error:
            print(error)   

    save_test_dataset(test_readers, reader_paths, test_feature_folder_name, dataset_path, audio_file_name)