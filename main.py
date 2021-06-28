from preprocessing import * 
from mel_spectrogram import * 
from model import *
from episode import *
from dataset_manager import *
from loss import *
import numpy
from tqdm import tqdm
import torch
import time

n_episodes = 60000
source_path = "./Dataset/English spoken wikipedia/english/"
audio_file_name = "audio.ogg"

if __name__ == "__main__":
    if torch.cuda.is_available():
      device = torch.device("cuda")
      print('Device:', torch.device('cuda'))
    else:
      device = torch.device("cpu")
    C = 2 # classes
    K = 1 # instances per class
    Q = 16 # query set size

    # To construct a C-way K-shot training episode, we randomly sample a reader from the training set, 
    # sample C word classes from the reader, and sample K instances per class as the support set.

    train_loss = []
    
    model = Protonet()
    if torch.cuda.is_available():
      model.cuda()
    print("Model parameters:",count_parameters(model))
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)

    for episode in tqdm(range(int(100)), desc = "episode"):
        query, support = batch_sample("Training_features/", C, K, Q)
        support = torch.FloatTensor(support)
        query = torch.FloatTensor(query)
        if torch.cuda.is_available():
          support = support.to(device='cuda')
          query = query.to(device='cuda')
        #support = torch.as_tensor(support, dtype = torch.float)
        #query = torch.as_tensor(query, dtype = torch.float)
        
        sample = {'xs' : support,    # support
                  'xq' : query}      # query
        
        model.train()
        optim.zero_grad()
        loss_out, output = loss(sample, model)
        #print(loss_out)
        #start = time.time()
        loss_out.backward()
        #print("loss_out.backward(x):",time.time() - start)
        optim.step()
        # TO DO: EARLY STOPPING
        train_loss.append(loss_out.item())
        
    print(train_loss)