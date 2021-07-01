import os
import numpy
import torch
from preprocessing import * 
from mel_spectrogram import * 
from tqdm import tqdm

def find_classes(reader, C, K, Q = 16):
    
    """
    find_classes returns classes, a list of dict that has 'word', 'start', 'end', 'folders' keys.
    'word' value is a string, the word name.
    'start', 'end', 'folders' values are lists. The i-th element of 'start' and 'end' value
    corresponds to the i-th element of 'folder' value.
    If the reader contains less words than C random sample the words, otherwise random sample 
    C words.

    Parameters:
    reader (string): one reader from the json file of readers 
    C (int): class size
    K (int): support set size
    Q (int): query set size (default: 16)

    Returns:
    classes (list): 
    """
    classes = []

    # taking at most C words from the reader
    # If the reader contains more words than C, random sample C words
    if (C <= len(reader['words'])):
        reader_words = random.sample(reader['words'], C)
    # If the reader contains less words than C, random sample the words
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

def save_dataset(readers, folder_name, dataset_path, audio_file_name, C, K, Q = 16):
    """
    save_dataset saves a pytorch tensor for each word of a reader in a folder named as the 
    reader name.

    Parameters:
    readers (string): the list of the readers
    folder_name (string): name of the folder in which save the features
    dataset_path (string): path of the Spoken Wikipedia Corpora dataset
    audio_file_name (string): name of the ".ogg" audio file 
    C (int): class size
    K (int): support set size
    Q (int): query set size (default: 16)

    Returns:
    """
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
                        # calculation of the center time of the word
                        word_center_time = (start_in_sec + end_in_sec)/2
                        # path of the audio file
                        audio_file_path = dataset_path + item['folders'][i] + "/" + audio_file_name
                        if (os.path.exists(audio_file_path)):
                            # compute the 128 bit log mel-spectrogram
                            item_spectrogram = compute_melspectrogram(audio_file_path, word_center_time)
                            # construction of a spectrogram tensor
                            spectrograms = np.concatenate((spectrograms, [item_spectrogram]), axis = 0)
                    # save the spectrograms tensor only if the first dimension is K + Q, that is
                    # when the word has K + Q instances
                    if (spectrograms.shape[0] == K + Q):
                        # conversion from numpy array to torch.FloatTensor
                        torch_tensor = torch.FloatTensor(spectrograms)
                        # save the torch tensor
                        torch.save(torch_tensor, folder_name + reader['reader_name'] + "/" + item['word'] + ".pt")
            except OSError as error:
                print(error)   

def batch_sample(feature_folder, C, K, Q = 16):
    """
    batch_sample returns the support and query set. 
    It reads each folder in feature_folder and take only the readers
    with at least C words. Next load the tensor with the 26 spectrograms of the word.
    Then random sample the instances (spectrograms) of the word. The first K spectrograms
    compose the support set and the last Q ones compose the query set.

    Parameters:
    feature_folder (string): the path of the features
    C (int): class size
    K (int): support set size
    Q (int): query set size (default: 16)

    Returns:
    support (torch.FloatTensor): support set
    query (torch.FloatTensor): query set
    """
    # initialize support tensor of dimension 0 x K x 128 x 51
    support =  torch.empty([0 ,K, 128, 51])
    # initialize query tensor of dimension 0 x Q x 128 x 51
    query = torch.empty([0, Q, 128, 51])
    reader_path = []
    # scan each reader folder in the feature_folder
    for entry in os.scandir(feature_folder):
        # create a list of reader names
        reader_path.append(entry.path)
        #print(entry.path)
    words = []
    # if a reader has less that C words, random sample another reader
    while (len(words) < C):
        # random sample a reader
        reader = random.sample(reader_path, 1)
        #print("folder:",folder[0])
        words = []
        # scan the torch tensor saved in each reader folder
        for word in os.scandir(reader[0]):
            # create a list containing the path of the words
            words.append(word.path)
    # random sample C paths of the words of a reader
    words = random.sample(words, C)
    # randomize the instances of each word
    for word in words:
        # load the tensor containing the spectrograms of the instances of one word
        spectrogram_buf = torch.load(word)
        # get the spectrogram tensor shape
        x_dim, y_dim, z_dim = spectrogram_buf.shape
        # get the number of instances
        instances_number = (spectrogram_buf.shape)[0]
        # rancom sample K + Q indices
        index = random.sample(list(torch.arange(instances_number)), K + Q)
        # initialize the spectrogram tensor
        spectrogram = torch.empty([0, 128, 51])
        for i in index:
            # concatenate spectrogram_buf with spectrogram to get a new tensor 
            # of random sampled instances of the word
            spectrogram = torch.cat((spectrogram, (spectrogram_buf[i, :, :]).view(1, y_dim, z_dim)), axis = 0)
        # concatenate the first K spectrograms with the support set
        support =  torch.cat((support, (spectrogram[:K]).view(1, K, y_dim, z_dim)), axis = 0)
        # concatenate the last Q spectrograms with the query set
        query = torch.cat((query, (spectrogram[K:K+Q]).view(1, Q, y_dim, z_dim)), axis = 0)
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

    #save_dataset(training_readers, training_feature_folder_name, dataset_path, audio_file_name, C, K, Q)
    
    if not (os.path.exists(validation_feature_folder_name)):
        try:
            os.mkdir(validation_feature_folder_name)
        except OSError as error:
            print(error) 

    #save_dataset(validation_readers, validation_feature_folder_name, dataset_path, audio_file_name, C, K, Q)