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
    print("n_query", n_query)

    target_inds = torch.tensor([1, 0])
    target_inds = target_inds.view(n_class, 1, 1).expand(n_class, int(n_query/2), 1).long()
    print(target_inds)
    #target_inds = Variable(target_inds, requires_grad=False)

    if torch.cuda.is_available():
        target_inds = target_inds.to(device='cuda')

    #print(target_inds)
    xs = torch.cat((positive_set, negative_set), 0)
    x = torch.cat((xs, query_set), 0)
    
    #print("x.shape",x.shape)4
    #start = time.time()
    embeddings = model(x)
    #print("model(x):",time.time() - start)
    #print("z.shape",z.shape)
    embeddings_dim = embeddings.size(-1)
    #print("z_dim",z_dim)
    positive_embeddings = embeddings[:p].view(1, p, embeddings_dim).mean(1)    
    negative_embeddings = embeddings[p:p+n].view(1, n, embeddings_dim).mean(1)   
    pos_neg_embeddings =  torch.cat((positive_embeddings, negative_embeddings), 0)
    query_embeddings = embeddings[p+n:]
    print("query_embeddings",query_embeddings.shape)
    print("pos_neg_embeddings",pos_neg_embeddings.shape)
    dists = euclidean_dist(query_embeddings, pos_neg_embeddings)
    print("dists",dists)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_class, int(n_query/2), -1)
    print('log_p_y', log_p_y)
    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    print('log_p_y.gather(2, target_inds)', log_p_y.gather(2, target_inds))
    print("loss_val", loss_val)

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    inverse_dist = torch.div(1.0, dists)
    prob = torch.log_softmax(inverse_dist, dim = 1)
    print("prob", prob)
    #prob = torch.log_softmax(-dists, dim = 1)

    return loss_val, prob, target_inds, {
        'loss': loss_val.item(),
        'acc': acc_val.item()
    }

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
    test_loss_values = []
    test_acc_values = []
    target_inds_iter = []

    for audio in tqdm(os.scandir("Test_features/"), desc = "Test features"):
        # getting the number of target keywords in each audio
        target_keywords_number = len([name for name in os.listdir(audio) if os.path.isfile(os.path.join(audio, name))])
        print(target_keywords_number)
        for i in range (target_keywords_number):
            for j in range (10):
                negative_set, positive_set, query_set = get_negative_positive_query_set(p, n, i, audio)  

                _, prob, target_inds, output = test_loss(model, negative_set, positive_set, query_set, n, p)

                test_loss_values.append(output['loss'])
                test_acc_values.append(output['acc'])

                '''  Probability array for positive class'''
                prob_pos = prob[:,1] # TO DO: 0 or 1?
                prob_pos = prob_pos.detach().cpu().tolist()
                #print('len(prob_pos)', len(prob_pos))
                #print('(prob_pos)', (prob_pos))
                prob_pos_iter.extend(prob_pos)
                target_inds = target_inds.reshape(-1).to(device='cpu')
                target_inds_iter.extend(target_inds)

    avg_test_loss = np.mean(test_loss_values)
    avg_test_acc = np.mean(test_acc_values)
    avg_prob = np.mean(np.array(prob_pos_iter),axis=0)
    #print('Average test loss: {}  Average test accuracy: {}'.format(avg_test_loss, avg_test_acc))

    #print('Average test prob: {}'.format(avg_prob))

    average_precision = sklearn.metrics.average_precision_score(np.array(target_inds_iter), prob_pos_iter)
    print('Average precision: {}'.format(average_precision))

if __name__ == "__main__":
    main()