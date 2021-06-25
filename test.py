from torch.nn.functional import embedding
from model import *
from loss import *
import numpy
import torch

C = 2   # number of classes
K = 10  # number of instances
Q = 16  # number of queries
model = Protonet()
#print(model)
'''
spectrogram1 = numpy.load("Features/wodup/elephant.npy")
spectrogram2 = numpy.load("Features/wodup/as.npy")
support =  numpy.stack((spectrogram1[:K], spectrogram2[:K]), axis = 0)
query = numpy.stack((spectrogram1[K:], spectrogram2[K:]), axis = 0)
'''
support =  numpy.empty([0 ,K, 128, 51])
query = numpy.empty([0, Q, 128, 51])
print("support.shape:",support.shape)
print("query.shape:",query.shape)

spectrogram = numpy.load("Features/wodup/elephant.npy")
print("spectrogram[:K].shape:",spectrogram[:K].shape)
support =  numpy.concatenate((support, [spectrogram[:K]]), axis = 0)
query = numpy.concatenate((query, [spectrogram[K:]]), axis = 0)
print("support.shape:",support.shape)
print("query.shape:",query.shape)

spectrogram = numpy.load("Features/wodup/as.npy")
support =  numpy.concatenate((support, [spectrogram[:K]]), axis = 0)
query = numpy.concatenate((query, [spectrogram[K:]]), axis = 0)
print("support.shape:",support.shape)
print("query.shape:",query.shape)

model.train()
optim = torch.optim.Adam(model.parameters(), lr = 0.001)
optim.zero_grad()
support = torch.FloatTensor(support)
query = torch.FloatTensor(query)
#print(support.shape)
sample = {'xs' : support,    # support
          'xq' : query}    # query

loss_out, output = loss(sample, model)
print(loss_out)
loss_out.backward()
optim.step()