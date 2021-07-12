import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
from relation_network import *
from utils import *

# Hyper Parameters
FEATURE_DIM = 192
RELATION_DIM = 8
CLASS_NUM = 5
SAMPLE_NUM_PER_CLASS = 5
BATCH_NUM_PER_CLASS = 16
EPISODE = 60000
TEST_EPISODE = 1000
LEARNING_RATE = 0.001
GPU = 0 # mettere a 1 in colab
HIDDEN_UNIT = 10

def main():
    print("Validation...")
    total_rewards = 0

    for i in range(TEST_EPISODE):
        degrees = random.choice([0,90,180,270])
        task = tg.OmniglotTask(metatest_character_folders,CLASS_NUM,SAMPLE_NUM_PER_CLASS,SAMPLE_NUM_PER_CLASS,)
        sample_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="train",shuffle=False,rotation=degrees)
        test_dataloader = tg.get_data_loader(task,num_per_class=SAMPLE_NUM_PER_CLASS,split="test",shuffle=True,rotation=degrees)

        sample_images,sample_labels = sample_dataloader.__iter__().next()
        test_images,test_labels = test_dataloader.__iter__().next()

        # calculate features
        sample_features = feature_encoder(Variable(sample_images).cuda(GPU)) # 5x64
        sample_features = sample_features.view(CLASS_NUM,SAMPLE_NUM_PER_CLASS,FEATURE_DIM,5,5)
        sample_features = torch.sum(sample_features,1).squeeze(1)
        test_features = feature_encoder(Variable(test_images).cuda(GPU)) # 20x64

        # calculate relations
        # each batch sample link to every samples to calculate relations
        # to form a 100x128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(SAMPLE_NUM_PER_CLASS*CLASS_NUM,1,1,1,1)
        test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM,1,1,1,1)
        test_features_ext = torch.transpose(test_features_ext,0,1)

        relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,FEATURE_DIM*2,5,5)
        relations = relation_network(relation_pairs).view(-1,CLASS_NUM)

        _,predict_labels = torch.max(relations.data,1)

        rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(CLASS_NUM*SAMPLE_NUM_PER_CLASS)]

        total_rewards += np.sum(rewards)

    test_accuracy = total_rewards/1.0/CLASS_NUM/SAMPLE_NUM_PER_CLASS/TEST_EPISODE

    print("test accuracy:",test_accuracy)

    if test_accuracy > last_accuracy:

        # save networks
        torch.save(feature_encoder.state_dict(),str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))
        torch.save(relation_network.state_dict(),str("./models/omniglot_relation_network_"+ str(CLASS_NUM) +"way_" + str(SAMPLE_NUM_PER_CLASS) +"shot.pkl"))

        print("save networks for episode:",episode)

        last_accuracy = test_accuracy
        
if __name__ == '__main__':
    main()