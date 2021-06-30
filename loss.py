#https://github.com/jakesnell/prototypical-networks/blob/c9bb4d258267c11cb6e23f0a19242d24ca98ad8a/protonets/models/few_shot.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.autograd import Variable

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def loss(sample, model):
    xs = Variable(sample['xs']) # support
    xq = Variable(sample['xq']) # query

    n_class = xs.size(0)
    assert xq.size(0) == n_class
    n_support = xs.size(1)
    n_query = xq.size(1)

    target_inds = torch.arange(0, n_class).view(n_class, 1, 1).expand(n_class, n_query, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)

    if xq.is_cuda:
        target_inds = target_inds.cuda()

    x = torch.cat([xs.view(n_class * n_support, *xs.size()[2:]),
                    xq.view(n_class * n_query, *xq.size()[2:])], 0)
    #print("x.shape",x.shape)4
    #start = time.time()
    z = model(x)
    #print("model(x):",time.time() - start)
    #print("z.shape",z.shape)
    z_dim = z.size(-1)
    #print("z_dim",z_dim)
    z_proto = z[:n_class*n_support].view(n_class, n_support, z_dim).mean(1)    
    #print("z_proto",z_proto.shape)
    zq = z[n_class*n_support:]
    #print("z_q",zq.shape)
    dists = euclidean_dist(zq, z_proto)
    #print("dists",dists.shape)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_class, n_query, -1)

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    _, y_hat = log_p_y.max(2)
    acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

    return loss_val, acc_val
    #{
        #'loss': loss_val.item(),
        #'acc': acc_val.item()
    #}