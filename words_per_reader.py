import json
import os

source_path = "./Dataset/English spoken wikipedia/english/"

f = open("readers_paths.json", "r")
readers = json.load(f)
f.close()

training_readers = []

for reader in readers:

    new_training_reader = {'reader_name' : reader['reader_name'],\
                           'words' : [] }

    # for every reader creates a new dict for words
    words_per_reader = [] 

    json_file_exist = False

    for reader_folder in reader['folder']:
        if (os.path.exists(source_path + "/" + reader_folder + "/word_count.json")):

            json_file_exist = True

            f = open(source_path + "/" + reader_folder + "/word_count.json", "r")
            recording_words = json.load(f)
            f.close()
            
            folder_per_word = []
            
            for word in recording_words:
                folder_per_word = {'folder' : reader_folder, \
                                 'start' : [word['start']], \
                                 'end' : [word['end']]}

                if not any (word['word'] == word_per_reader['word'] for word_per_reader in words_per_reader):
                   words_per_reader.append({ 'word' : word['word'], \
                                             'folders' : [folder_per_word]}) \

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
                                
    if (json_file_exist):
        new_training_reader['words'] = words_per_reader
        training_readers.append(new_training_reader)

f = open("readers_words.json","w")
f.write(json.dumps(training_readers, indent = 0, sort_keys = False))
f.close()

