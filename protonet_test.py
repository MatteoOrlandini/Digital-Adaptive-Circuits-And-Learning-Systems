import torch
import os
import random
import numpy as np
from protonet import *
from utils import *
from protonet_loss import *
from tqdm import tqdm
import sklearn.metrics

def test_loss(model, negative_set, positive_set, query_set, n, p):
    n_class = 2
    n_support = n + p
    n_query = query_set.size(0)
    #print("n_query", n_query)

    target_inds = torch.tensor([1, 0])
    target_inds = target_inds.view(n_class, 1, 1).expand(n_class, int(n_query/2), 1).long()
    target_inds = target_inds.reshape(-1)
    #print(target_inds)
    #target_inds = Variable(target_inds, requires_grad=False)

    if torch.cuda.is_available():
        target_inds = target_inds.to(device='cuda')

    #print(target_inds)
    xs = torch.cat((positive_set, negative_set), 0)   
    #print("xs.shape", xs.shape)     
    #print("xs", xs)  
    x = torch.cat((xs, query_set), 0)
    #print("x.shape", x.shape)  
    #print("x", x)  
    
    embeddings = model(x)
    #print("embeddings", embeddings)
    embeddings_dim = embeddings.size(-1)
    #print("embeddings.shape", embeddings.shape)
    #print("embeddings_dim", embeddings_dim)
    
    #print("embeddings_sum", torch.sum(embeddings))
    
    positive_embeddings = embeddings[:p].view(1, p, embeddings_dim).mean(1)    
    #print("embeddings[:p]", embeddings[:p])  
    #print("positive_embeddings", positive_embeddings)
    negative_embeddings = embeddings[p:p+n].view(1, n, embeddings_dim).mean(1)     
    #print("embeddings[p:p+n]", embeddings[p:p+n])
    #print("negative_embeddings", negative_embeddings)
    pos_neg_embeddings =  torch.cat((positive_embeddings, negative_embeddings), 0)
    #print("pos_neg_embeddings", pos_neg_embeddings)
    query_embeddings = embeddings[p+n:]
    #print("query_embeddings",query_embeddings)
    dists = euclidean_dist(query_embeddings, pos_neg_embeddings)
    #print("dists",dists)

    #inverse_dist = torch.div(1.0, dists)
    prob = torch.log_softmax(-dists, dim = 1)

    return prob, target_inds

def get_negative_positive_query_set(p, n, i, audio):
    # initialize support tensor of dimension p x 128 x 51
    positive_set = torch.empty([0, 128, 51])
    negative_set = torch.empty([0, 128, 51])
    # initialize query tensor of dimension n x 128 x 51
    query_set = torch.empty([0, 128, 51])
    
    words = []
    for word in os.scandir(audio.path):
        words.append(word.path)
        
    #pos_word = random.sample(words, 1)
    pos_word = words[i]

    spectrograms = torch.load(pos_word)
    #print('spectrograms shape', spectrograms.shape)
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
    """
    print('negative_set', negative_set.shape)
    print('positive_set', positive_set.shape)
    print('query_set', query_set.shape)
    """
    return negative_set, positive_set, query_set

def main():
    p = 5
    n = 10

    C = 2
    K = 1

    model = Protonet()
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)

    if torch.cuda.is_available():
        checkpoint = torch.load("Models/Prototypical/prototypical_model_C{}_K{}_60000epi.pt".format(C, K), map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load("Models/Prototypical/prototypical_model_C{}_K{}_60000epi.pt".format(C, K), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])


    model.eval()

    if torch.cuda.is_available():
        model.to(device='cuda')

    prob_pos_iter = []
    target_inds_iter = []

    for audio in tqdm(os.scandir("Test_features/"), desc = "Test features"):
        # getting the number of target keywords in each audio
        target_keywords_number = len([name for name in os.listdir(audio) if os.path.isfile(os.path.join(audio, name))])

        for i in range (target_keywords_number):
            for j in range (1):
                negative_set, positive_set, query_set = get_negative_positive_query_set(p, n, i, audio)  

                prob, target_inds = test_loss(model, negative_set, positive_set, query_set, n, p)

                '''  Probability array for positive class'''
                prob_pos = prob[:,0] 
                prob_pos = prob_pos.detach().cpu().tolist()
                #print('len(prob_pos)', len(prob_pos))
                #print('(prob_pos)', (prob_pos))
                prob_pos_iter.extend(prob_pos)
                #print("prob_pos_iter",prob_pos_iter)
                target_inds = target_inds.to(device='cpu')
                target_inds_iter.extend(target_inds)
                #print("target_inds",np.array(target_inds_iter))

    avg_prob = np.mean(np.array(prob_pos_iter),axis=0)
    print('Average test prob: {}'.format(avg_prob))

    average_precision = sklearn.metrics.average_precision_score(np.array(target_inds_iter), prob_pos_iter)
    print('Average precision: {}'.format(average_precision))

if __name__ == "__main__":
    main()