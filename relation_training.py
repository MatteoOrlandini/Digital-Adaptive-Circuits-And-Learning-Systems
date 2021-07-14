#-------------------------------------
# Project: Learning to Compare: Relation Network for Few-Shot Learning
# Date: 2017.9.21
# Author: Flood Sung
# All Rights Reserved
#-------------------------------------


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import numpy as np
import argparse
from relation_network import *
from utils import *
from tqdm import tqdm, trange

"""
parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 64)
parser.add_argument("-r","--relation_dim",type = int, default = 8)
parser.add_argument("-w","--class_num",type = int, default = 5)
parser.add_argument("-s","--sample_num_per_class",type = int, default = 5)
parser.add_argument("-b","--batch_num_per_class",type = int, default = 15)
parser.add_argument("-e","--episode",type = int, default= 1000000)
parser.add_argument("-t","--test_episode", type = int, default = 1000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
args = parser.parse_args() 
"""


# Hyper Parameters
FEATURE_DIM = 64
RELATION_DIM = 8
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 5
BATCH_NUM_PER_CLASS = 16
EPISODE = 60000
TEST_EPISODE = 1000
LEARNING_RATE = 0.001
GPU = 0
HIDDEN_UNIT = 10

def main():
    # Step 1: init data folders
    print("init data folders")
    # init character folders for dataset construction
    training_readers, validation_readers = get_training_validation_readers("Training_validation_features/", CLASS_NUM)

    # Step 2: init neural networks
    print("init neural networks")

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    feature_encoder.apply(weights_init)
    relation_network.apply(weights_init)

    if torch.cuda.is_available():
        feature_encoder.to(device='cuda')
        relation_network.to(device='cuda')

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr = LEARNING_RATE)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr = LEARNING_RATE)

    # Step 3: build graph
    print("Training...")
    
    train_loss = []

    last_accuracy = 0.0

    for episode in trange(EPISODE, desc = "training episode", position = 0, leave = True):

        # sample datas
        batches, samples = batch_sample(training_readers, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS) # samples: C X K X 128 X 51,  batches: C X Q X 128 X 51
        
        samples = samples.view(CLASS_NUM * SAMPLE_NUM_PER_CLASS, 1, *samples.size()[2:])  # (C X K) X 1 X 51 X 51
        batches = batches.view(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, *batches.size()[2:])   # (C X Q) X 1 X 51 X 51

        if torch.cuda.is_available():
            samples = samples.to(device='cuda')
            batches = batches.to(device='cuda')

        # calculate features
        sample_features = feature_encoder(Variable(samples)) # (CLASS_NUM * SAMPLE_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5

        # resize the images from 15 X 5 to 5 X 5 to get square images
        # interpolate down samples the input to the given size
        sample_features = F.interpolate(sample_features, size = 5)
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5) # CLASS_NUM X SAMPLE_NUM_PER_CLASS X FEATURE_DIM X 5 X 5
        sample_features = torch.sum(sample_features,1).squeeze(1)   # CLASS_NUM X FEATURE_DIM X 5 X 5

        batch_features = feature_encoder(Variable(batches)) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
        batch_features = F.interpolate(batch_features, size = 5)

        # calculate relations
        # each batch sample link to every samples to calculate relations
        sample_features_ext = sample_features.unsqueeze(0).repeat(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 1, 1, 1) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)  # 5 X (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5

        relation_pairs = torch.cat((sample_features_ext,batch_features_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X (FEATURE_DIM * 2) X 5 X 5
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM) #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X CLASS_NUM

        mse = nn.MSELoss()
        
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM))

        for i in range(CLASS_NUM):
            for j in range(BATCH_NUM_PER_CLASS):
                one_hot_labels[BATCH_NUM_PER_CLASS*i+j,i] = 1

        if torch.cuda.is_available():
            mse = mse.to(device='cuda')
            one_hot_labels = one_hot_labels.to(device='cuda')

        loss = mse(relations,one_hot_labels)

        # training

        feature_encoder.zero_grad()
        relation_network.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)

        feature_encoder_optim.step()
        relation_network_optim.step()

        train_loss.append(loss.item())

        # validation

        if (episode+1)%5000 == 0:
            total_rewards = 0
            #for i in trange(TEST_EPISODE, desc = "validation episode", position = 1, leave = True):
            for i in range(TEST_EPISODE):
                # sample datas
                batches, samples = batch_sample(validation_readers, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS) # samples: C X K X 128 X 51,  batches: C X Q X 128 X 51
                
                samples = samples.view(CLASS_NUM * SAMPLE_NUM_PER_CLASS, 1, *samples.size()[2:])  # (C X K) X 1 X 51 X 51
                batches = batches.view(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, *batches.size()[2:])   # (C X Q) X 1 X 51 X 51

                if torch.cuda.is_available():
                    samples = samples.to(device='cuda')
                    batches = batches.to(device='cuda')

                # calculate features
                sample_features = feature_encoder(Variable(samples)) # (CLASS_NUM * SAMPLE_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5

                # resize the images from 15 X 5 to 5 X 5 to get square images
                # interpolate down samples the input to the given size
                sample_features = F.interpolate(sample_features, size = 5)
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5) # CLASS_NUM X SAMPLE_NUM_PER_CLASS X FEATURE_DIM X 5 X 5
                sample_features = torch.sum(sample_features,1).squeeze(1)   # CLASS_NUM X FEATURE_DIM X 5 X 5

                batch_features = feature_encoder(Variable(batches)) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
                batch_features = F.interpolate(batch_features, size = 5)

                # calculate relations
                # each batch sample link to every samples to calculate relations
                sample_features_ext = sample_features.unsqueeze(0).repeat(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 1, 1, 1) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5
                batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)  # 5 X (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
                batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5

                relation_pairs = torch.cat((sample_features_ext,batch_features_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X (FEATURE_DIM * 2) X 5 X 5
                relations = relation_network(relation_pairs).view(-1,CLASS_NUM) #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X CLASS_NUM

                _,predict_labels = torch.max(relations.data,1)

                test_labels = torch.arange(CLASS_NUM).expand(BATCH_NUM_PER_CLASS,CLASS_NUM).transpose(1,0).reshape(-1)

                rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM*BATCH_NUM_PER_CLASS)]

                total_rewards += np.sum(rewards)

                '''
                mse = nn.MSELoss()
                
                one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM))

                for i in range(CLASS_NUM):
                    for j in range(BATCH_NUM_PER_CLASS):
                        one_hot_labels[BATCH_NUM_PER_CLASS*i+j,i] = 1

                if torch.cuda.is_available():
                    mse = mse.to(device='cuda')
                    one_hot_labels = one_hot_labels.to(device='cuda')

                loss = mse(relations,one_hot_labels)

                feature_encoder_optim.step()
                relation_network_optim.step()

                validation_loss.append(loss.item())
                '''
            test_accuracy = total_rewards/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS/TEST_EPISODE

            print("test accuracy:",test_accuracy)

            if test_accuracy > last_accuracy:
                torch.save({
                            'epoch': episode+1,
                            'feature_encoder_state_dict': feature_encoder.state_dict(),
                            'relation_network_state_dict' : relation_network.state_dict(),
                            'feature_encoder_optim_state_dict': feature_encoder_optim.state_dict(),
                            'relation_network_optim_state_dict': relation_network_optim.state_dict(),
                            'loss': train_loss,
                            'avg_loss_tr' : np.mean(train_loss),
                            'valid_accuracy' : test_accuracy,
                            }, "Models/Relation/relation_model_C{}_K{}.pt".format(CLASS_NUM, SAMPLE_NUM_PER_CLASS))
                last_accuracy = test_accuracy
            
if __name__ == '__main__':
        main()