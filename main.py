from preprocessing import * 
from mel_spectrogram import * 
from model import *
from episode import *
from dataset_manager import *
from model import *
from loss import *
import numpy
from tqdm import tqdm
import torch

n_episodes = 60000
source_path = "./Dataset/English spoken wikipedia/english/"
audio_file_name = "audio.ogg"

if __name__ == "__main__":
    C = 10 # classes
    K = 10 # instances per class
    Q = 16 # query set size

    # To construct a C-way K-shot training episode, we randomly sample a reader from the training set, 
    # sample C word classes from the reader, and sample K instances per class as the support set.

    train_loss = []
    
    model = Protonet()
    #print("Model parameters:",count_parameters(model))
    optim = torch.optim.Adam(model.parameters(), lr = 0.001)

    for episode in tqdm(range(int(10)), desc = "episode"):
        query, support = batch_sample("Training_features/", C, K, Q)
        support = torch.FloatTensor(support)
        query = torch.FloatTensor(query)
        
        sample = {'xs' : support,    # support
                  'xq' : query}      # query
        
        model.train()
        optim.zero_grad()
        loss_out, output = loss(sample, model)
        #print(loss_out)
        loss_out.backward()
        optim.step()
        # TO DO: EARLY STOPPING
        train_loss.append(loss_out.item())
        
    print(train_loss)