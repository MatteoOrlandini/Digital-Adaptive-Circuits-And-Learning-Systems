import os
import numpy
from preprocessing import * 
from mel_spectrogram import * 
from tqdm import tqdm

def save_dataset(readers, C, K, Q, folder_name, dataset_path, audio_file_name):
    for reader in tqdm(readers, position = 0):
        if not (os.path.exists(folder_name + reader['reader_name'])):
            try:
                os.mkdir(folder_name + reader['reader_name'])
                classes = find_classes(reader, C, K, Q)
                for item in tqdm(classes, position = 1):
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
                        numpy.save(folder_name + reader['reader_name'] + "/" + item['word'] + ".npy", spectrograms)
            except OSError as error:
                print(error)   

def extract_feature(feature_folder, C, K, Q):
    support =  numpy.empty([0 ,K, 128, 51])
    query = numpy.empty([0, Q, 128, 51])
    #print("support.shape:",support.shape)
    #print("query.shape:",query.shape)
    reader_path = []
    for entry in os.scandir(feature_folder):
        reader_path.append(entry.path)
        #print(entry.path)
    folder = random.sample(reader_path, 1)
    #print("folder:",folder[0])
    words = []
    for word in os.scandir(folder[0]):
        words.append(word.path)
    words = random.sample(words, C)
    #print("words:",words)
    for word in words:
        spectrogram = numpy.load(word)
        #print("numpy.shape(spectrogram):",numpy.shape(spectrogram))
        len = numpy.shape(spectrogram)[0]
        index = random.sample(list(numpy.arange(len)), K + Q)
        spectrogram_buf = numpy.empty([0, 128, 51])
        for i in index:
            spectrogram_buf = numpy.concatenate((spectrogram_buf, [spectrogram[i, :, :]]), axis = 0)
        support =  numpy.concatenate((support, [spectrogram_buf[:K]]), axis = 0)
        query = numpy.concatenate((query, [spectrogram_buf[K:K+Q]]), axis = 0)
        #print("support.shape:",support.shape)
        #print("query.shape:",query.shape)
    return query, support


if __name__ == "__main__":
    C = 10 # classes
    K = 10 # instances per class
    Q = 16
    feature_folder_name = "Features/"
    dataset_path = "./Dataset/English spoken wikipedia/english/"
    audio_file_name = "audio.ogg"
    valid_readers = find_valid_readers(C, K, Q)
    if not (os.path.exists(feature_folder_name)):
        try:
            os.mkdir(feature_folder_name)
        except OSError as error:
            print(error)   
    save_dataset(valid_readers, C, K, Q, feature_folder_name, dataset_path, audio_file_name)

    
    