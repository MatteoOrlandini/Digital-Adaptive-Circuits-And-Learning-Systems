from preprocessing import * 
from mel_spectrogram import * 
from model import Protonet
from episode import *
import torch
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import time

n_episodes = 60000
source_path = "./Dataset/English spoken wikipedia/english/"
audio_file_name = "audio.ogg"

if __name__ == "__main__":
    C = 2 # classes
    K = 1 # instances per class
    Q = 16 # query set size
    valid_readers = find_valid_readers(C, K, Q)

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
    
    # To construct a C-way K-shot training episode, we randomly sample a reader from the training set, 
    # sample C word classes from the reader, and sample K instances per class as the support set.

    train_loss = []

    model = Protonet()
    optim = torch.optim.Adam(model.parameters(),lr=0.001)

    for episode_number in range(3):
        x = torch.empty([0, 128, 51])  #features, torch.empty returns a new array of shape (128, 51)
        y = torch.empty(0) #labels
        training_reader = random.sample(training_readers, 1)
        training_classes = find_classes(training_reader[0], C, K, Q)
        for idx, item in enumerate (training_classes):
            print('word:', item['word'])
            for i in range(len(item['start'])):
                idx = torch.tensor([idx])
                y = torch.cat((y,idx), axis = 0)
                start_in_sec = item['start'][i]/1000 # conversion from milliseconds to seconds
                end_in_sec = item['end'][i]/1000    # conversion from milliseconds to seconds
                word_center_time = (start_in_sec + end_in_sec)/2
                #start = time.time()
                item_spectrogram = compute_melspectrogram(source_path + item['folders'][i] + "/" + audio_file_name, \
                                                    word_center_time)
                #print('item_spectrogram time:', time.time() - start)
                #print(item_spectrogram.shape)
                x = torch.cat((x, torch.tensor([item_spectrogram])), axis = 0)
                print('x.shape:', x.shape)
                #print('y.shape:', y.shape)

        loss, y_pred = proto_net_episode(model, optim, x, y, K, C, Q, True)
        # da: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/train.py
        train_loss.append(loss.item())

    print(train_loss)

"""     fig, ax = plt.subplots()
    img = librosa.display.specshow(x[3], x_axis = "ms", y_axis = "mel", sr = 16000, hop_length = 160, ax = ax)
    ax.set(title = 'Mel spectrogram display')
    fig.colorbar(img, ax = ax)
    plt.show() """