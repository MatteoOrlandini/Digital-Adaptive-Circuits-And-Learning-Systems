from protonet import *
from utils import *
from protonet_loss import *
import torch
from tqdm import tqdm, trange
import numpy as np

def main():
  C = 2 # classes
  K = 5 # instances per class
  Q = 16 # query set size

  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: {}".format(device))
    print("Device name: {}".format(torch.cuda.get_device_properties(device).name))
  else:
    device = torch.device("cpu")

  # To construct a C-way K-shot training episode, we randomly sample a reader from the training set, 
  # sample C word classes from the reader, and sample K instances per class as the support set.

  model = Protonet()
  if torch.cuda.is_available():
    model.to(device='cuda')

  print("Model parameters: {}".format(count_parameters(model)))

  optim = torch.optim.Adam(model.parameters(), lr = 0.001)

  training_readers, validation_readers = get_training_validation_readers("Training_validation_features/", C)

  train_loss = []
  train_acc = []

  for episode in trange(60000, desc = "episode", position = 0, leave = True):
      query, support = batch_sample(training_readers, C, K, Q)
      if torch.cuda.is_available():
        support = support.to(device='cuda')
        query = query.to(device='cuda')
      
      model.train()
      optim.zero_grad()
      loss_out, acc_val = loss(support, query, model, "euclidean_dist")

      print('before', loss_out.is_cuda)
      if torch.cuda.is_available():
        loss_out = loss_out.to(device='cuda')
      print('after', loss_out.is_cuda)

      loss_out.backward()
      optim.step()
      # TO DO: EARLY STOPPING
      train_loss.append(loss_out.item())
      train_acc.append(acc_val.item())
      
  print("Training loss: {}".format(train_loss))
  print("Training accuracy: {}".format(train_acc))

  avg_loss_tr = np.mean(train_loss)
  avg_acc_tr = np.mean(train_acc)
  print('Average train loss: {}  Average training accuracy: {}'.format(avg_loss_tr,avg_acc_tr))

  torch.save({
              'epoch': 60000,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optim.state_dict(),
              'loss': train_loss,
              'acc' : train_acc,
              'avg_loss_tr' : np.mean(train_loss),
              'avg_acc_tr' : np.mean(train_acc),
              }, "Models/model_C{}_K{}_60000epi.pt".format(C, K))

if __name__ == '__main__':
    main()