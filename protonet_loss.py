#https://github.com/jakesnell/prototypical-networks/blob/c9bb4d258267c11cb6e23f0a19242d24ca98ad8a/protonets/models/few_shot.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def euclidean_dist(x, y):
    """ 
    euclidean_dist computes the euclidean distance from two arrays x and y

    Parameters:
    x (torch.FloatTensor): query embeddings 
    y (torch.FloatTensor): prototype embeddings

    Returns:
    torch.pow(x - y, 2).sum(2) (double): the euclidean distance from x and y
    """
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cosine_dist(query, support):
    """
    cosine_dist returns cosine similarity between query and support, computed along dim = 0.
    
    Parameters:
    query (torch.FloatTensor): a query from the query set 
    support (torch.FloatTensor): a support from the support set 

    Returns:
    cos(query, support) (double): the cosine distance from query and support
    """
    cos = nn.CosineSimilarity(dim = 0)
    return cos(query, support)

def loss(xs, xq, model, distance_type):
    """
    loss returns the loss and accuracy value. It calculates p_y, the loss 
    softmax over distances to the prototypes in the embedding space. We need to 
    minimize the negative log-probability of p_y to proceed the learning process.

    Parameters:
    xs (torch.FloatTensor): support set
    xq (torch.FloatTensor): query set
    model (torch.nn.Module): neural network model
    distance_type (string): "euclidean_dist" or "cosine_dist"

    Returns:
    loss_val (double): loss value
    acc_val (double): accuracy value
    """
    # xs = Variable()
    n_class = xs.size(0)
    assert xq.size(0) == n_class
    n_support = xs.size(1)
    n_query = xq.size(1)

    x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                    xq.view(n_class * n_query, *xq.size()[2:])], 0)
                    
    embeddings = model(x)
    
    embeddings_dim = embeddings.size(-1)
    
    if (distance_type == "euclidean_dist"):
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
        print("target inds", target_inds)
        print("euclidean target inds shape", target_inds.shape)
        prototypes = embeddings[:n_class*n_support].view(n_class, n_support, embeddings_dim).mean(1)  
        queries = embeddings[n_class*n_support:]
        dists = euclidean_dist(queries, prototypes)
        print("dists",dists)
        print("euclidean dists.shape", dists.shape)

        if torch.cuda.is_available():
            target_inds = target_inds.to(device='cuda')

        log_p_y = F.log_softmax(-dists, dim = 1).view(n_class, n_query, -1)
        print('log_p_y.shape', log_p_y.shape)
        print('log_p_y', log_p_y)
        
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        print('target_inds)', target_inds)
        print('log_p_y.gather(2, target_inds)', log_p_y.gather(2, target_inds))
        print('log_p_y.gather(2, target_inds).shape', log_p_y.gather(2, target_inds).shape)

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    elif (distance_type == "cosine_dist"):
        target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query*n_support, 1).long()
        #print("targ ind", target_inds)
        #print("cosine target_inds shape", target_inds.shape)
        supports = embeddings[:n_class*n_support]
        queries = embeddings[n_class*n_support:]
        dists = torch.empty(n_query*n_support*n_class, 0)
        for i in range(n_class):
            #distances = torch.empty(n_query*n_support)
            distances = torch.empty(0)
            for k in range(n_query):
                for j in range(n_support*n_class):
                    cosine_distance = cosine_dist(queries[i*n_query+k], supports[j])
                    #andare a capo a metà e poi concatenare
                    distances = torch.cat((distances, torch.tensor([cosine_distance], requires_grad = True)), 0)
            #distances = torch.tensor(distances)
            distances = distances.view(-1,1) 
            print("dists.shape:",dists.shape)
            #print("distances.shape:",distances.shape)
            dists = torch.cat((dists, distances), 1)
        #dists=torch.transpose(dists, 0, 1)
        print("dists",dists)
        #print("cosine dist shape", dists.shape)

        if torch.cuda.is_available():
            target_inds = target_inds.to(device='cuda')

        #print("dists", dists.shape)
        log_p_y = F.log_softmax(-dists, dim = 1).view(n_class, n_support*n_query, -1)
        print('log_p_y', log_p_y)
        print('log_p_y.shape', log_p_y.shape)
        
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        print('target_inds)', target_inds)
        print('log_p_y.gather(2, target_inds)', log_p_y.gather(2, target_inds))
        print('log_p_y.gather(2, target_inds).shape', log_p_y.gather(2, target_inds).shape)

        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()
    else:
        raise ValueError("Please use distance_type = \"euclidean_dist\" or distance_type == \"cosine_dist\"")
    

    
    return loss_val, acc_val