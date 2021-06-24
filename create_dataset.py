import numpy
from preprocessing import * 
from mel_spectrogram import * 

source_path = "./Dataset/English spoken wikipedia/english/"
audio_file_name = "audio.ogg"

if __name__ == "__main__":
    C = 2 # classes
    K = 1 # instances per class
    Q = 16
    valid_readers = find_valid_readers(C, K, Q)
    print('len(valid_readers):', len(valid_readers))
    for valid_reader in valid_readers:
        print('valid_reader:', valid_reader['reader_name'])
        classes = find_classes(valid_reader)
        for item in classes:
            #print('word:', item['word'])
            for i in range(len(item['start'])):
                start_in_sec = item['start'][i]/1000 # conversion from milliseconds to seconds
                end_in_sec = item['end'][i]/1000    # conversion from milliseconds to seconds
                word_center_time = (start_in_sec + end_in_sec)/2
                item_spectrogram = compute_melspectrogram(source_path + item['folders'][i] + "/" + audio_file_name, \
                                        word_center_time)