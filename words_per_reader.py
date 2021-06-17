import json
import os


f = open("readers_paths.json", "r")
readers = json.load(f)
f.close()

training_readers = []

for reader in readers:

    new_training_reader = {'reader_name' : reader['reader_name'],\
                           'words' : [] }

    # for every reader creates a new dict for words
    words_per_reader = [] 

    for reader_path in reader['paths']:
        if (os.path.exists(reader_path + "/word_count.json")):

            f = open(reader_path + "/word_count.json", "r")
            recording_words = json.load(f)
            f.close()
            
            for word in recording_words:
                path_per_word = {'path' : reader_path, \
                                 'start' : [word['start']], \
                                 'end' : [word['end']]}

                if not any (word['word'] == word_per_reader['word'] for word_per_reader in words_per_reader):
                   words_per_reader.append({ 'word' : word['word'], \
                                             'paths' : [path_per_word]}) \

                else:
                    for word_per_reader in words_per_reader:
                        if word['word']==word_per_reader['word']:
                            if reader_path in word_per_reader['paths']:
                                for path in word_per_reader['paths']:
                                    if reader_path == path['path']:
                                        path['start'] += word['start']
                                        path['end'] += word['end']
                            else:
                                word_per_reader['paths'].append(path_per_word)

f = open("readers_words.json","w")
f.write(json.dumps(training_readers, indent = None, sort_keys = False))
f.close()

