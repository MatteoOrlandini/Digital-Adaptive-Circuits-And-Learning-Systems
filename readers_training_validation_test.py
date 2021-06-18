import json
import random

def write_json_file(filename, list_of_dict):
    f = open(filename, "w")
    f.write(json.dumps(list_of_dict, indent = 4))
    f.close

def find_valid_readers(C, K):
    f = open("readers_words.json", "r")
    readers = json.load(f)
    f.close()

    valid_readers = []

    Q = 16 # query set

    for reader in readers:    
        valid_words = []
        number_of_words_per_reader = 0
        is_valid_reader = False
        number_of_valid_words = 0

        for word in reader['words']:
            start = []
            end = []
            folders = []

            for item in word['folders']:
                start += item['start']
                end += item['end']
                folders += [item['folder']]*len(item['start'])

            new_word = {'word'  : word['word'], \
                        'start' : start, \
                        'end'   : end, \
                        'folder': folders}

            if (len(new_word['start']) >= K + Q):
                number_of_valid_words += 1
                valid_words.append(new_word)

        if (number_of_valid_words >= C):
            is_valid_reader = True

        if (is_valid_reader):
            valid_readers.append({'reader_name' : reader['reader_name'],\
                                    'words': valid_words})

    return valid_readers

def create_training_validation_test_readers(valid_readers, number_of_training_readers, number_of_test_readers, number_of_validation_readers):
    training_readers = []
    test_and_validation_readers = []
    test_readers = []
    validation_readers = []

    training_readers = random.sample(valid_readers, number_of_training_readers)

    for item in valid_readers:
        if not any(item['reader_name'] == training_reader['reader_name'] for training_reader in training_readers):
            test_and_validation_readers.append(item)

    test_readers = test_and_validation_readers[0:number_of_test_readers]

    validation_readers = test_and_validation_readers[number_of_test_readers:number_of_test_readers+number_of_validation_readers]

    write_json_file("training_readers.json", training_readers)
    write_json_file("test_readers.json", test_readers)
    write_json_file("validation_readers.json", validation_readers)


if __name__ == "__main__":
    C = 2 # classes
    K = 10 # instances per class
    valid_readers = find_valid_readers(C, K)

    if (len(valid_readers) >= 183):
        number_of_training_readers = 138
        number_of_test_readers = 30
        number_of_validation_readers = 15

    else:
        number_of_training_readers = int(138/183*len(valid_readers))
        number_of_test_readers = int(30/183*len(valid_readers))
        number_of_validation_readers = int(15/183*len(valid_readers))

    create_training_validation_test_readers(valid_readers, number_of_training_readers, number_of_test_readers, number_of_validation_readers)