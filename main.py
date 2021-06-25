from preprocessing import * 
from mel_spectrogram import * 
from model import *
from episode import *
import torch
import random
import numpy as np
import time
from model import *
from loss import *
import numpy
from tqdm import tqdm
import torch
import os
from dataset_manager import *

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
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)

    for episode in tqdm(range(int(10)), desc = "episode"):
        query, support = extract_feature("Features/", C, K, Q)

        support = torch.FloatTensor(support)
        query = torch.FloatTensor(query)
        #print(support.shape)
        sample = {'xs' : support,    # support
                  'xq' : query}    # query
        
        model.train()
        optim.zero_grad()
        loss_out, output = loss(sample, model)
        #print(loss_out)
        loss_out.backward()
        optim.step()
        # TO DO: EARLY STOPPING
        train_loss.append(loss_out.item())

    print(train_loss)