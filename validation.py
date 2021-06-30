import numpy as np
import torch
from tqdm import tqdm
from model import *
from loss import *
from dataset_manager import *

model = Protonet()
optim = torch.optim.Adam(model.parameters(), lr = 0.001)

C = 2
K = 1
Q = 16

if torch.cuda.is_available():
    checkpoint = torch.load("model_C" + str(C) + "_K" + str(K) + "_60000epi.pt", map_location=torch.device('cuda'))
else:
    checkpoint = torch.load("model_C" + str(C) + "_K" + str(K) + "_60000epi.pt", map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
optim.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
train_loss = checkpoint['loss']
train_acc = checkpoint['acc']
avg_loss_tr = checkpoint['avg_loss_tr']
avg_acc_tr = checkpoint['avg_acc_tr']

avg_loss_tr = np.mean(train_loss)
avg_acc_tr = np.mean(train_acc)
print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr,avg_acc_tr))

if torch.cuda.is_available():
  device = torch.device("cuda")
  print('Device:', torch.device('cuda'))
else:
  device = torch.device("cpu")

model.eval()

if torch.cuda.is_available():
  model.cuda()

valid_loss = []
valid_acc = []

for episode in tqdm(range(int(60000)), desc = "episode"):
    query, support = batch_sample("Validation_features/", C, K, Q)
    support = torch.FloatTensor(support)
    query = torch.FloatTensor(query)
    if torch.cuda.is_available():
      support = support.to(device='cuda')
      query = query.to(device='cuda')
    #support = torch.as_tensor(support, dtype = torch.float)
    #query = torch.as_tensor(query, dtype = torch.float)
    
    sample = {'xs' : support,    # support
              'xq' : query}      # query
    
    val_loss, acc_val = loss(sample, model)
    #print("loss_out.backward(x):",time.time() - start)
    optim.step()
    # TO DO: EARLY STOPPING
    valid_loss.append(val_loss.item())
    valid_acc.append(acc_val.item())

print("Validation loss:", valid_loss)
print("Validation accuracy:", valid_acc)

avg_loss_val = np.mean(valid_loss)
avg_acc_val = np.mean(valid_acc)
print('Average validation loss: {}  Average validation accuracy: {}'.format(avg_loss_val, avg_acc_val))

torch.save({
            'epoch': 60000,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'train_loss': train_loss,
            'train_acc' : train_acc,
            'avg_loss_tr' : np.mean(train_loss),
            'avg_acc_tr' : np.mean(train_acc),
            'valid_loss': valid_loss,
            'valid_acc' : valid_acc,
            'avg_loss_val' : np.mean(valid_loss),
            'avg_acc_val' : np.mean(valid_acc),
            }, "model_valid_C" + str(C) + "_K" + str(K) + "_60000epi.pt")