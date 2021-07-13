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
    #metatrain_character_folders,metatest_character_folders = tg.omniglot_character_folders()

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
    #feature_encoder_scheduler = StepLR(feature_encoder_optim,step_size=100000,gamma=0.5)
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr = LEARNING_RATE)
    #relation_network_scheduler = StepLR(relation_network_optim,step_size=100000,gamma=0.5)

    '''
    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load feature encoder success")
    if os.path.exists(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl")))
        print("load relation network success")
    '''

    # Step 3: build graph
    print("Training...")

    last_accuracy = 0.0

    for episode in trange(EPISODE, desc = "episode", position = 0, leave = True):

        #feature_encoder_scheduler.step(episode)
        #relation_network_scheduler.step(episode)

        # init dataset
        # sample_dataloader is to obtain previous samples for compare
        # batch_dataloader is to batch samples for training
        '''
        degrees = random.choice([0,90,180,270])
        task = tg.OmniglotTask(metatrain_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,BATCH_NUM_PER_CLASS)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        batch_dataloader = tg.get_data_loader(task,num_per_class=BATCH_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)
        '''

        batches, samples = batch_sample(training_readers, CLASS_NUM, SAMPLE_NUM_PER_CLASS, BATCH_NUM_PER_CLASS) # samples: C X K X 128 X 51,  batches: C X Q X 128 X 51
        #print(batches.shape)
        # resize the images from * X 128 X 51 to * X 51 X 51 to get square images
        # interpolate down samples the input to the given size
        samples = F.interpolate(samples, size = 51)
        batches = F.interpolate(batches, size = 51)
        #print(batches.shape)
        samples = samples.view(CLASS_NUM * SAMPLE_NUM_PER_CLASS, 1, *samples.size()[2:])  # (C X K) X 1 X 51 X 51
        batches = batches.view(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, *batches.size()[2:])   # (C X Q) X 1 X 51 X 51

        if torch.cuda.is_available():
            samples = samples.to(device='cuda')
            batches = batches.to(device='cuda')

        # sample datas
        #samples,sample_labels = sample_dataloader.__iter__().next()
        #batches,batch_labels = batch_dataloader.__iter__().next()
        #batch_labels = torch.arange(0, CLASS_NUM).view(CLASS_NUM, 1, 1).expand(CLASS_NUM, BATCH_NUM_PER_CLASS, 1).long()
        batch_labels = torch.arange(0, CLASS_NUM)   # [0, 1, 2, ..., CLASS_NUM]
        # calculate features
        sample_features = feature_encoder(Variable(samples)) # (CLASS_NUM * SAMPLE_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
        #print("sample_features.shape:",sample_features.shape)
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5) # CLASS_NUM X SAMPLE_NUM_PER_CLASS X FEATURE_DIM X 5 X 5
        sample_features = torch.sum(sample_features,1).squeeze(1)
        #print("sample_features.shape after view and sum:", sample_features.shape)   # CLASS_NUM X FEATURE_DIM X 5 X 5

        batch_features = feature_encoder(Variable(batches)) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
        #print("batch_features.shape:", batch_features.shape)

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(CLASS_NUM * BATCH_NUM_PER_CLASS, 1, 1, 1, 1) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5
        #print("sample_features_ext.shape after repeat:", sample_features_ext.shape)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)  # 5 X (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
        #print("batch_features_ext.shape after repeat:", batch_features_ext.shape)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5
        #print("batch_features_ext.shape after transpose:", batch_features_ext.shape)

        relation_pairs = torch.cat((sample_features_ext,batch_features_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X (FEATURE_DIM * 2) X 5 X 5
        #print("relation_pairs.shape :", relation_pairs.shape)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM) #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X CLASS_NUM
        #print("relations.shape :", relations.shape)

        mse = nn.MSELoss()
        one_hot_labels = Variable(torch.zeros(BATCH_NUM_PER_CLASS*CLASS_NUM, CLASS_NUM).scatter_(1, batch_labels.view(-1,1), 1))

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

        if (episode+1)%100 == 0:
                print("episode:",episode+1,"loss",loss.data[0])

if __name__ == '__main__':
    main()