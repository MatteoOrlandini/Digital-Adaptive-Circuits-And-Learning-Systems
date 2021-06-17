import json
import os


f = open("readers_paths.json", "r")
readers = json.load(f)
f.close()

training_readers = []

for reader in readers:
    #reader['words']=[{'word':"frar",'start':[1,2],'end':[3,4],'path':["dadsa","cacdc"]}]
    #reader['words'] = []
    
    #training_readers.append({'reader_name' : reader['reader_name'],\
    #                         'words' : [] })

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
                                            #{'path' : reader_path, \
                                            # 'timestamp':\
                                            #{'start' : [word['start']], \
                                            # 'end' : [word['end']]}}]})
                else:
                    for word_per_reader in words_per_reader:
                        word_per_reader

                #    word_per_reader['paths'].append(path_per_word)

                #for word_per_reader in words_per_reader:
                    #if (reader_path not in word_per_reader['paths']['path']):
                        #word_per_reader['paths'].append(reader_path)
                        #word_per_reader['paths']['path']['timestamp'] = word['start']
                        #word_per_reader['paths']['path']['timestamp'] = word['end']
            
            new_training_reader['words'] = words_per_reader
            training_readers.append(new_training_reader)
'''
                words_reader_array.append(word['word'])

            unique_words = []
            for word_readers in words_reader_array:
                if word_readers not in unique_words:
                    unique_words.append({'word' : word_readers ,\
                                        'paths' : []})

            new_training_reader['words'] = unique_words

            # creating paths for every words
            for reader_path in reader['paths']:
                if (os.path.exists(reader_path + "/word_count.json")):
                    
                    f = open(reader_path + "/word_count.json", "r")
                    recording_words = json.load(f)
                    f.close()
            
            training_readers.append(new_training_reader)
'''
'''
            for training_readers_words in training_readers['words']:
                training_readers[len(training_readers)]['words']['word'].append(word['word'])
                

            if not any(new_reader['reader_name'] == reader['reader_name'] for new_reader in new_readers):
                new_readers = [{'reader_name' : reader['reader_name'],'words':'prova'}]


            print("path esiste!")
            for word in words:
                #print("parola")
                #if not any(reader['reader_name'] == reader_name for reader in readers):
                for x in reader['words']:
                    print("sono dentro l'array")
                    if word['word'] not in x['word']:
                        print("la parola è nuova")
                        reader['words'].append({'word':word['word'],'start':word['start'],'end':word['end'],'path':[path]*word['frequency']})
                    else:
                        print("la parola non è nuova")
                        for y in reader['words']:
                            for item in y['word']:
                                if word['word'] == item:
                                    item['start'] += word['start']
                                    item['end'] += word['end']
                                    item['path'] += [path]*word['frequency']
'''
f = open("readers_words.json","w")
f.write(json.dumps(training_readers, indent = 4, sort_keys = False))
f.close()

