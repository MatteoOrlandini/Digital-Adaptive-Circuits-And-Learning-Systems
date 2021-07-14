from protonet import *
from utils import *
from protonet_loss import *
import torch
from tqdm import tqdm, trange
import numpy as np
import scipy.io

def main():
  C = 2 # classes
  K = 1 # instances per class
  Q = 16 # query set size

  if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Device: {}".format(device))
    print("Device name: {}".format(torch.cuda.get_device_properties(device).name))
  else:
    device = torch.device("cpu")
    
  model = Protonet()
  if torch.cuda.is_available():
    model.to(device='cuda')

  print("Model parameters: {}".format(count_parameters(model)))

  optim = torch.optim.Adam(model.parameters(), lr = 0.001)

  training_readers, validation_readers = get_training_validation_readers("Training_validation_features/", C)

  print ("Training...")

  model.train()

  last_accuracy = 0.0
  train_loss = []
  train_acc = []

  # To construct a C-way K-shot training episode, we randomly sample a reader from the training set, 
  # sample C word classes from the reader, and sample K instances per class as the support set.

  for episode in trange(60000, desc = "episode", position = 0, leave = True):
      query, support = batch_sample(training_readers, C, K, Q)
      if torch.cuda.is_available():
        support = support.to(device='cuda')
        query = query.to(device='cuda')
      
      optim.zero_grad()
      loss_out, acc_val = loss(support, query, model)

      loss_out.backward()
      optim.step()
      
      train_loss.append(loss_out.item())
      train_acc.append(acc_val.item())
      
      if (episode+1)%5000 == 0:
        valid_loss = []
        valid_acc = []
        print ("\nValidation...")

        model.eval()

        for validation_episode in range(1000):
            query, support = batch_sample(validation_readers, C, K, Q)
            if torch.cuda.is_available():
              support = support.to(device='cuda')
              query = query.to(device='cuda')
            
            val_loss, acc_val = loss(support, query, model)
            optim.step()
            
            valid_loss.append(val_loss.item())
            valid_acc.append(acc_val.item())

        model.train()

        print("\nValidation accuracy: {}".format(np.mean(valid_acc)))

        if np.mean(valid_acc) > last_accuracy: 

          torch.save({
                'epoch': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'train_loss': train_loss,
                'train_acc' : train_acc,              
                'valid_loss': valid_loss,
                'valid_acc' : valid_acc,
                'avg_loss_tr' : np.mean(train_loss),
                'avg_acc_tr' : np.mean(train_acc),
                'avg_loss_val' : np.mean(valid_loss),
                'avg_acc_val' : np.mean(valid_acc),
                }, "/Models/Prototypical/prototypical_model_C{}_K{}.pt".format(C, K))
          
          last_accuracy = np.mean(valid_acc)

          scipy.io.savemat('/Models/Prototypical/prototypical_results_C{}_K{}.mat'.format(C, K), {'train_loss': train_loss , 'train_acc' : train_acc, 'valid_loss':valid_loss,'valid_acc':valid_acc})

if __name__ == '__main__':
    main()