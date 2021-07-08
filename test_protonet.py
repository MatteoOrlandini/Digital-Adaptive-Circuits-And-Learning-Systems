import torch
import os
import random
import numpy as np
from model import *
from loss import *
from tqdm import tqdm
import sklearn.metrics

p = 5
n = 10

C = 2
K = 1

test_loss = []
test_acc = []
prob_list = []
target_inds_iter = []

model = Protonet()
optim = torch.optim.Adam(model.parameters(), lr = 0.001)

if torch.cuda.is_available():
    checkpoint = torch.load("Models/model_C{}_K{}_60000epi.pt".format(C, K), map_location=torch.device('cuda'))
else:
    checkpoint = torch.load("Models/model_C{}_K{}_60000epi.pt".format(C, K), map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])

prob_pos_iter = []
for audio in tqdm(os.scandir("Test_features/"), desc = "Test features"):
    # initialize support tensor of dimension p x 128 x 51
    positive_set =  torch.empty([0, 128, 51])
    negative_set =  torch.empty([0, 128, 51])
    # initialize query tensor of dimension n x 128 x 51
    query_set = torch.empty([0, 128, 51])
    
    words = []
    for word in os.scandir(audio.path):
        words.append(word.path)
        
    pos_word = random.sample(words, 1)

    spectrograms = torch.load(pos_word[0])
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

    words.remove(pos_word[0])

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

    model.eval()

    if torch.cuda.is_available():
        model.to(device='cuda')

    n_class = 2
    n_support = n + p
    n_query = query_set.size(0)

    target_inds = torch.tensor([1, 0])
    target_inds = target_inds.view(n_class, 1, 1).expand(n_class, int(n_query/2), 1).long()
    #print(target_inds)
    #target_inds = Variable(target_inds, requires_grad=False)

    if torch.cuda.is_available():
        target_inds = target_inds.to(device='cuda')

    #print(target_inds)
    xs = torch.cat((positive_set, negative_set), 0)
    x = torch.cat((xs, query_set), 0)
    
    #print("x.shape",x.shape)4
    #start = time.time()
    z = model(x)
    #print("model(x):",time.time() - start)
    #print("z.shape",z.shape)
    z_dim = z.size(-1)
    #print("z_dim",z_dim)
    z_proto_p = z[:p].view(1, p, z_dim).mean(1)    
    z_proto_n = z[p:p+n].view(1, n, z_dim).mean(1)   
    z_proto =  torch.cat((z_proto_p, z_proto_n), 0)
    zq = z[p+n:]
    #print("z_q",zq.shape)
    dists = euclidean_dist(zq, z_proto)
    #print("dists",dists)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)
    #print('log_p_y', log_p_y)
    loss_val = -log_p_y.gather(1, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(1)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    inverse_dist = torch.div(1.0, dists)
    prob = torch.softmax(inverse_dist, dim=1)
    '''  Probability array for positive class'''
    prob_pos = prob[:,0]
    prob_pos = prob_pos.detach().cpu().tolist()
    #print('len(prob_pos)', len(prob_pos))
    #print('(prob_pos)', (prob_pos))
    test_loss.append(loss_val.item())
    test_acc.append(acc_val.item())
    prob_pos_iter.extend(prob_pos)
    target_inds = target_inds.reshape(-1).to(device='cpu')
    target_inds_iter.extend(target_inds)
    #print(target_inds)
    #print("shape(target_inds)", target_inds.shape)
    #print(prob_pos)

#print("Test loss: {}".format(test_loss))
#print("Test accuracy: {}".format(test_acc))

avg_test_loss = np.mean(test_loss)
avg_test_acc = np.mean(test_acc)
avg_prob = np.mean(np.array(prob_pos_iter),axis=0)
print('Average test loss: {}  Average test accuracy: {}'.format(avg_test_loss, avg_test_acc))

print('Average test prob: {}'.format(avg_prob))

average_precision = sklearn.metrics.average_precision_score(np.array(target_inds_iter), prob_pos_iter)
print('Average precision: {}'.format(average_precision))