import os
import numpy
from preprocessing import * 
from mel_spectrogram import * 
from tqdm import tqdm

def save_dataset(readers, C, K, Q, folder_name, dataset_path, audio_file_name):
    for reader in tqdm(readers):
        if not (os.path.exists(folder_name + reader['reader_name'])):
            try:
                os.mkdir(folder_name + reader['reader_name'])
                classes = find_classes(reader, C, K, Q)
                for instance in tqdm(classes):
                    spectrograms = np.empty([0, 128, 51])
                    for i in range(len(instance['start'])):
                        start_in_sec = instance['start'][i]/1000 # conversion from milliseconds to seconds
                        end_in_sec = instance['end'][i]/1000     # conversion from milliseconds to seconds
                        word_center_time = (start_in_sec + end_in_sec)/2
                        audio_file_path = dataset_path + instance['folders'][i] + "/" + audio_file_name
                        if (os.path.exists(audio_file_path)):
                            item_spectrogram = compute_melspectrogram(audio_file_path, word_center_time)
                            spectrograms = np.concatenate((spectrograms, [item_spectrogram]), axis = 0)
                    if (spectrograms.shape[0] == C):
                        numpy.save(folder_name + reader['reader_name'] + "/" + instance['word'] + ".npy", spectrograms)
            except OSError as error:
                print(error)   

def load_dataset(feature_folder_name):
    for path in os.scandir(feature_folder_name):
        reader = os.path.basename(path)
        spectrogram = numpy.load(path + "/" + reader)
        return spectrogram


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

    for path in os.scandir(feature_folder_name):
        reader = os.path.basename(path)
        spectrogram = numpy.load(path + "/" + reader)

    
    