from tqdm import tqdm
import os
from json_manager import *

source_path = "./Dataset/English spoken wikipedia/english/"
# read the json that contains the readers name and their audio folders
readers = read_json_file("readers_paths.json")
#initialize a list of dict
readers_words = []
# for each reader search the word spoken by the reader
for reader in tqdm(readers):
    # create a dict with 'reader_name' and 'words' keys
    new_readers_words = {'reader_name' : reader['reader_name'],\
                         'words' : [] }

    # for each reader create a new dict for words
    words_per_reader = [] 
    # flag to signal if "word_count.json" exists
    json_file_exist = False
    # search for each folder the file "word_count.json"
    for reader_folder in reader['folder']:
        if (os.path.exists(source_path + "/" + reader_folder + "/word_count.json")):
            # "word_count.json" exists
            json_file_exist = True
            # read "word_count.json"
            recording_words = read_json_file(source_path + "/" + reader_folder + "/word_count.json")
            # for each audio folder create a new list of dict
            folder_per_word = []
            # for each word in the audio save the folder, start and end timestamps
            for word in recording_words:
                 # create a dict with 'folder', 'start' and 'end' keys
                folder_per_word = {'folder' : reader_folder, \
                                   'start' : word['start'], \
                                   'end' : word['end']}
                # if the word is not yet in the list add the word
                if not any (word['word'] == word_per_reader['word'] for word_per_reader in words_per_reader):
                   words_per_reader.append({ 'word' : word['word'], \
                                             'folders' : [folder_per_word]}) \
                # if the word is already in the list add 'start' and 'end' if the "reader_folder" is 
                # in word_per_reader['folders'] or add the "folder_per_word"  
                else:
                    for word_per_reader in words_per_reader:
                        if word['word'] == word_per_reader['word']:
                            if reader_folder in word_per_reader['folders']:
                                for folder in word_per_reader['folders']:
                                    if reader_folder == folder['folder']:
                                        folder['start'] += word['start']
                                        folder['end'] += word['end']
                            else:
                                word_per_reader['folders'].append(folder_per_word)
    # if "word_count.json" exists add the new reader to "training_readers" list     
    if (json_file_exist):
        new_readers_words['words'] = words_per_reader
        readers_words.append(new_readers_words)
# save a "readers_words.json" with the name of the readers and the relative words spoken
write_json_file("readers_words.json", readers_words, indent = 0)
