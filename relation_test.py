import torch
import os
import random
import numpy as np
from relation_network import *
from utils import *
from protonet_loss import *
from tqdm import tqdm
import sklearn.metrics

def test_predictions(feature_encoder,relation_network, positive_set, negative_set, query_set, n, p):
    n_class = 2
    n_query = query_set.size(0)

    FEATURE_DIM = 64

    pos_embeddings = feature_encoder(Variable(positive_set)) # (CLASS_NUM * SAMPLE_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5

    # resize the images from 15 X 5 to 5 X 5 to get square images
    # interpolate down samples the input to the given size
    pos_embeddings = F.interpolate(pos_embeddings, size = 5)
    pos_embeddings = pos_embeddings.view(1, p, FEATURE_DIM, 5, 5) # CLASS_NUM X SAMPLE_NUM_PER_CLASS X FEATURE_DIM X 5 X 5
    pos_embeddings = torch.sum(pos_embeddings,1).squeeze(1)   # CLASS_NUM X FEATURE_DIM X 5 X 5

    neg_embeddings = feature_encoder(Variable(negative_set)) # (CLASS_NUM * SAMPLE_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5

    # resize the images from 15 X 5 to 5 X 5 to get square images
    # interpolate down samples the input to the given size
    neg_embeddings = F.interpolate(neg_embeddings, size = 5)
    neg_embeddings = neg_embeddings.view(1, n, FEATURE_DIM, 5, 5) # CLASS_NUM X SAMPLE_NUM_PER_CLASS X FEATURE_DIM X 5 X 5
    neg_embeddings = torch.sum(neg_embeddings,1).squeeze(1)   # CLASS_NUM X FEATURE_DIM X 5 X 5
    
    batch_features = feature_encoder(Variable(query_set)) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
    batch_features = F.interpolate(batch_features, size = 5)
    
    pos_neg_embeddings =  torch.cat((pos_embeddings, neg_embeddings), 0)

    # calculate relations
    # each batch sample link to every samples to calculate relations
    pos_neg_embeddings_ext = pos_neg_embeddings.unsqueeze(0).repeat(int(n_class * n_query/2), 1, 1, 1, 1) # (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5
    
    batch_features_ext = batch_features.unsqueeze(0).repeat(n_class, 1, 1, 1, 1)  # 5 X (CLASS_NUM * BATCH_NUM_PER_CLASS) X FEATURE_DIM X 5 X 5
    batch_features_ext = torch.transpose(batch_features_ext, 0, 1)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X 5 X FEATURE_DIM X 5 X 5

    relation_pairs = torch.cat((pos_neg_embeddings_ext,batch_features_ext), 2).view(-1, FEATURE_DIM*2, 5, 5)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X (FEATURE_DIM * 2) X 5 X 5
    relations = relation_network(relation_pairs).view(-1,n_class) #  (CLASS_NUM * BATCH_NUM_PER_CLASS) X CLASS_NUM
    
    target_inds = torch.arange(n_class-1, -1, step = -1).view(n_class, 1).expand(n_class, int(n_query/2)).long()
    
    return relations[:,0].view(-1).detach(), target_inds.reshape(1, -1).squeeze()

def get_negative_positive_query_set(p, n, i, audio):
    # initialize support tensor of dimension p x 128 x 51
    positive_set = torch.empty([0, 128, 51])
    negative_set = torch.empty([0, 128, 51])
    # initialize query tensor of dimension n x 128 x 51
    query_set = torch.empty([0, 128, 51])
    
    words = []
    for word in os.scandir(audio.path):
        words.append(word.path)
        
    pos_word = words[i]

    spectrograms = torch.load(pos_word)
    index = np.arange(spectrograms.shape[0])
    pos_index = random.sample(list(index), p)
    for i in pos_index:
        pos = spectrograms[i, :, :]
        positive_set = torch.cat((positive_set, pos.view(1, 128, 51)), axis = 0)
    for i in index:
        if i not in pos_index:
            query = spectrograms[i, :, :]
            query_set = torch.cat((query_set, query.view(1, 128, 51)), axis = 0)

    query_label = [1]*query_set.shape[0]
    query_label += [0]*query_set.shape[0]

    words.remove(pos_word)

    for i in range(n):
        neg = random.sample(words, 1)
        spectrograms = torch.load(neg[0])
        index = np.arange(spectrograms.shape[0])
        neg_index = random.sample(list(index), 1)
        neg = spectrograms[neg_index, :, :]
        negative_set = torch.cat((negative_set, neg.view(1, 128, 51)), axis = 0)

    for i in range(query_set.shape[0]):
        query_sample = random.sample(words, 1)
        spectrograms = torch.load(query_sample[0])
        index = np.arange(spectrograms.shape[0])
        query_index = random.sample(list(index), 1)
        query = spectrograms[query_index, :, :]
        query_set = torch.cat((query_set, query.view(1, 128, 51)), axis = 0)
        
    return negative_set, positive_set, query_set

def main():
    p = 5
    n = 10

    C = 2
    K = 5

    FEATURE_DIM = 64
    RELATION_DIM = 8

    feature_encoder = CNNEncoder()
    relation_network = RelationNetwork(FEATURE_DIM, RELATION_DIM)

    if torch.cuda.is_available():
        checkpoint = torch.load("Models/Relation/relation_model_C{}_K{}.pt".format(C, K), map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load("Models/Relation/relation_model_C{}_K{}.pt".format(C, K), map_location=torch.device('cpu'))
    feature_encoder.load_state_dict(checkpoint['feature_encoder_state_dict'])
    relation_network.load_state_dict(checkpoint['relation_network_state_dict'])


    feature_encoder.eval()
    relation_network.eval()

    if torch.cuda.is_available():
        feature_encoder.to(device='cuda')
        relation_network.to(device='cuda')

    auc_list = []
    for audio in tqdm(os.scandir("Test_features/"), desc = "Test features"):
        y_pred = []
        y_true = []
        # getting the number of target keywords in each audio
        target_keywords_number = len([name for name in os.listdir(audio) if os.path.isfile(os.path.join(audio, name))])

        for i in range (target_keywords_number):
            for j in range (1):
                negative_set, positive_set, query_set = get_negative_positive_query_set(p, n, i, audio)

                negative_set = negative_set.view(1 * n, 1, *negative_set.size()[1:])  
                positive_set = positive_set.view(1 * p, 1, *positive_set.size()[1:]) 
                query_set = query_set.view(int(C * query_set.size()[0]/2), 1, *query_set.size()[1:])   # (C X Q) X 1 X 51 X 51
                
                y_pred_tmp, y_true_tmp = test_predictions(feature_encoder,relation_network, positive_set, negative_set, query_set, n, p)

                y_pred.extend(y_pred_tmp)
                y_true.extend(y_true_tmp)

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
        auc_tmp = sklearn.metrics.auc(recall, precision)
        print("auc_tmp: {}".format(auc_tmp))
        auc_list.append(auc_tmp)
    """
    prob_query_pos_iter.extend(prob_query_neg_iter)
    prob_pos_iter = prob_query_pos_iter

    target_inds_iter = np.ones(int(len(prob_pos_iter)/2))
    target_inds_iter = np.append(target_inds_iter, np.zeros(int(len(prob_pos_iter)/2)))

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(target_inds_iter, prob_pos_iter)
    """
    auc = np.mean(auc_list)
    print("Area under precision recall curve: {}".format(auc))
    auc_std_dev = np.std(auc_list)
    print("Standard deviation area under precision recall curve: {}".format(auc_std_dev))

    #average_precision = sklearn.metrics.average_precision_score(np.array(target_inds_iter), prob_pos_iter)
    #print('Average precision: {}'.format(average_precision))

if __name__ == "__main__":
    main()