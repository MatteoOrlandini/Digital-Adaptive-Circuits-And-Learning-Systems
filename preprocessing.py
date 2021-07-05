import random
import numpy
from json_manager import *

def find_valid_readers(C, K, Q = 16):
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

            # if there are at least K + Q instances append a new valid word
            if (len(new_word['start']) >= K + Q):
                number_of_valid_words += 1
                valid_words.append(new_word)

        # if the number of valid words is at least C, then the reader is valid
        if (number_of_valid_words >= C):
            is_valid_reader = True

        # if the reader is valid then add the reader to the valid readers list
        if (is_valid_reader):
            valid_readers.append({'reader_name' : reader['reader_name'],\
                                    'words': valid_words})

    #write_json_file("valid_readers.json", valid_readers)
    return valid_readers

def create_training_validation_test_readers(valid_readers, number_of_training_readers, number_of_test_readers, number_of_validation_readers):

    # take the first ("number_of_training_readers") "valid_readers" to create the training readers
    training_readers = valid_readers[0 : number_of_training_readers]

    # take from ("number_of_training_readers") index to ("number_of_training_readers" + "number_of_test_readers") index
    # of "valid_readers" to create the test readers
    test_readers = valid_readers[number_of_training_readers : number_of_training_readers + number_of_test_readers]

    # take from ("number_of_training_readers" + "number_of_test_readers") index to 
    # ("number_of_training_readers" + "number_of_test_readers" + "number_of_validation_readers") index
    # of "valid_readers" to create the validation readers
    validation_readers = valid_readers[number_of_training_readers + number_of_test_readers : \
                                                     number_of_training_readers + number_of_test_readers + number_of_validation_readers]

    #write_json_file("training_readers.json", training_readers)
    #write_json_file("test_readers.json", test_readers)
    #write_json_file("validation_readers.json", validation_readers)

    return training_readers, test_readers, validation_readers

if __name__ == "__main__":
    C = 2 # classes
    K = 10 # instances per class
    Q = 16
    
    valid_readers = find_valid_readers(C, K, Q)
    
# The readers are partitioned into training, validation, and test sets with a 138:15:30 ratio
    number_of_training_readers = int(138/183*len(valid_readers))
    number_of_test_readers = int(30/183*len(valid_readers))
    number_of_validation_readers = int(15/183*len(valid_readers))

    # The valid readers are partitioned into training, validation, and test readers
    training_readers, test_readers, validation_readers = create_training_validation_test_readers(valid_readers, \
                                                                                                number_of_training_readers, \
                                                                                                number_of_test_readers, \
                                                                                                number_of_validation_readers)
    