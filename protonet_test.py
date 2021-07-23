import torch
import os
import random
import numpy as np
from protonet import *
from utils import *
from protonet_loss import *
from tqdm import tqdm
import sklearn.metrics
import matplotlib.pyplot as plt

def test_predictions(model, negative_set, positive_set, query_set, n, p):
    n_class = 2
    n_query = query_set.size(0)
    
    xs = torch.cat((positive_set, negative_set), 0)   
    x = torch.cat((xs, query_set), 0)
    
    embeddings = model(x)
    embeddings_dim = embeddings.size(-1)
    
    positive_embeddings = embeddings[:p].view(1, p, embeddings_dim).mean(1)    
    negative_embeddings = embeddings[p:p+n].view(1, n, embeddings_dim).mean(1)     
    pos_neg_embeddings =  torch.cat((positive_embeddings, negative_embeddings), 0)
    query_embeddings = embeddings[p+n:]
    dists = euclidean_dist(query_embeddings, pos_neg_embeddings)

    target_inds = torch.arange(n_class-1, -1, step = -1).view(n_class, 1).expand(n_class, int(n_query/2)).long()

    p_y = F.softmax(-dists, dim = 1).view(n_class, int(n_query/2), -1)

    return p_y[:,:,0].view(-1).detach(), target_inds.reshape(1, -1).squeeze()

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
    K = 1

    model = Protonet()
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)

    if torch.cuda.is_available():
        checkpoint = torch.load("Models/Prototypical/prototypical_model_C{}_K{}.pt".format(C, K), map_location=torch.device('cuda'))
    else:
        checkpoint = torch.load("Models/Prototypical/prototypical_model_C{}_K{}.pt".format(C, K), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optim.load_state_dict(checkpoint['optimizer_state_dict'])


    model.eval()

    if torch.cuda.is_available():
        model.to(device='cuda')

    auc_list = []
    for audio in tqdm(os.scandir("Test_features/"), desc = "Test features"):
        y_pred = []
        y_true = []
        # getting the number of target keywords in each audio
        target_keywords_number = len([name for name in os.listdir(audio) if os.path.isfile(os.path.join(audio, name))])

        for i in range (target_keywords_number):
            for j in range (10):
                negative_set, positive_set, query_set = get_negative_positive_query_set(p, n, i, audio)  
                    
                if torch.cuda.is_available():
                  positive_set = positive_set.to(device='cuda')
                  negative_set = negative_set.to(device='cuda')
                  query_set = query_set.to(device='cuda')

                y_pred_tmp, y_true_tmp = test_predictions(model, negative_set, positive_set, query_set, n, p)
                y_pred_tmp = y_pred_tmp.cpu().tolist()
                y_true_tmp = y_true_tmp.cpu().tolist()
                
                y_pred.extend(y_pred_tmp)
                y_true.extend(y_true_tmp)
                """
                precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)
                fig, ax = plt.subplots()
                pr_display = sklearn.metrics.PrecisionRecallDisplay(precision, recall)
                pr_display.plot(ax=ax)
                plt.savefig('prec_rec_curves/pre_rec_{}.png'.format(i*10+j))
                plt.close()
                """
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_pred)

        auc_tmp = sklearn.metrics.auc(recall, precision)
        auc_list.append(auc_tmp)

    auc = np.mean(auc_list)
    print("Area under precision recall curve: {}".format(auc))
    auc_std_dev = np.std(auc_list)
    print("Standard deviation area under precision recall curve: {}".format(auc_std_dev))

    
if __name__ == "__main__":
    main()