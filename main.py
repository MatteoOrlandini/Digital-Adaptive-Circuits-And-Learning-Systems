import preprocessing as pre
import mel_spectrogram





if __name__ == "__main__":
    C = 10 # classes
    K = 10 # instances per class
    valid_readers = pre.find_valid_readers(C, K)

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
    training_readers, test_readers, validation_readers = pre.create_training_validation_test_readers(valid_readers, \
                                                                                                number_of_training_readers, \
                                                                                                number_of_test_readers, \
                                                                                                number_of_validation_readers)
    
    # To construct a C-way K-shot training episode, we randomly sample a reader from the training set, 
    # sample C word classes from the reader, and sample K instances per class as the support set.
    for training_reader in training_readers:
        training_classes = pre.find_classes(training_reader, C, K)
        for class in training_classes:
            mels = 
