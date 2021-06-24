from torch.nn.functional import embedding
from model import Protonet
import numpy
import torch
from torch.optim import Optimizer

model = Protonet()
#print(model)
spectrogram1 = numpy.load("Features/wodup/elephant.npy")
spectrogram2 = numpy.load("Features/wodup/as.npy")
features =  numpy.concatenate((spectrogram1[:10], spectrogram2[:10], spectrogram1[10:], spectrogram2[10:]), axis = 0)
print(features.shape)
model.train()
optim = torch.optim.Adam(model.parameters(),lr=0.001)
optim.zero_grad()
x = torch.FloatTensor(features)
print(x.shape)
embeddings = model(x)
print('embeddings:',embeddings.shape)
support = embeddings[:2*10]
queries = embeddings[2*10:]

distances = (
            queries.unsqueeze(1).expand(queries.shape[0], support.shape[0], -1) -
            support.unsqueeze(0).expand(queries.shape[0], support.shape[0], -1)
    ).pow(2).sum(dim=2)
    
print(distances.shape)
print(distances)