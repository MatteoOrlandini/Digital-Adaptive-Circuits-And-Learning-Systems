import json
import random
import numpy

def write_json_file(filename, list_of_dict):
    f = open(filename, "w")
    f.write(json.dumps(list_of_dict, indent = 4))
    f.close

def read_json_file(filename):
    f = open(filename, "r")
    list_of_dict = json.load(f)
    f.close
    return list_of_dict

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

    return valid_readers

def create_training_validation_test_readers(valid_readers, number_of_training_readers, number_of_test_readers, number_of_validation_readers):

    # random sample the valid readers the take a maximum of 
    # ("number_of_training_readers" + "number_of_test_readers" + "number_of_validation_readers") valid readers
    valid_readers = random.sample(valid_readers, number_of_training_readers + number_of_test_readers + number_of_validation_readers)

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

def find_classes(reader, C = None, K = None, Q = 16):
    classes = []

    # taking only C words from all training readers ones
    if not (C == None and K ==  None):
        reader_words = random.sample(reader['words'], C)  
    else:
        reader_words = reader['words']

    for word in reader_words:
        # numpy.arange returns evenly spaced values within a given interval.
        # create an array of index to get the start, end and folder of the same index
        index_array = list(numpy.arange(0, len(word['start']), step = 1))

        # random sample only K + Q indexes
        if not (C == None and K ==  None):
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

    #write_json_file("Classi/training_words_of_"+ training_reader['reader_name'] +".json", training_classes)
    return classes

""" if __name__ == "__main__":
    C = 10 # classes
    K = 10 # instances per class
    valid_readers = find_valid_readers(C, K)

    # The readers are partitioned into training, validation, and test sets with a 138:15:30 ratio
    if (len(valid_readers) >= 183):
        number_of_training_readers = 138
        number_of_test_readers = 30
        number_of_validation_readers = 15

    else:
        number_of_training_readers = int(138/183*len(valid_readers))
        number_of_test_readers = int(30/183*len(valid_readers))
        number_of_validation_readers = int(15/183*len(valid_readers))

    # The valid readers are partitioned into training, validation, and test readers
    training_readers, test_readers, validation_readers = create_training_validation_test_readers(valid_readers, \
                                                                                                number_of_training_readers, \
                                                                                                number_of_test_readers, \
                                                                                                number_of_validation_readers)

    
    for training_reader in training_readers:
        training_classes = find_classes(training_reader, C, K)
    # TO DO: find query and support set """