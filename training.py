from model import *
import os
from loss import *
import random
import torch
from tqdm import tqdm, trange
import numpy as np

def batch_sample(features, C, K, Q = 16):
    """
    batch_sample returns the support and query set. 
    It reads each folder in feature_folder and  load the tensor with the spectrograms of the word.
    Then random sample the instances (spectrograms) of the word to get only K+Q spectrograms. 
    The first K spectrograms compose the support set and the last Q ones compose the query set.

    Parameters:
    features (list): training/validation features paths
    C (int): class size
    K (int): support set size
    Q (int): query set size (default: 16)

    Returns:
    support (torch.FloatTensor): support set
    query (torch.FloatTensor): query set
    """
    # initialize support tensor of dimension 0 x K x 128 x 51
    support =  torch.empty([0, K, 128, 51])
    # initialize query tensor of dimension 0 x Q x 128 x 51
    query = torch.empty([0, Q, 128, 51])
    # random sample a reader
    reader = random.sample(features, 1)[0]
    words = []
    # scan the torch tensor saved in each reader folder
    for word in os.scandir(reader):
        # create a list containing the path of the words
        words.append(word.path)
    # random sample C paths of the words of a reader
    words = random.sample(words, C)
    # randomize the instances of each word
    for word in words:
        # load the tensor containing the spectrograms of the instances of one word
        spectrogram_buf = torch.load(word)
        # get the spectrogram tensor shape
        x_dim, y_dim, z_dim = spectrogram_buf.shape
        # get the number of instances
        instances_number = (spectrogram_buf.shape)[0]
        # rancom sample K + Q indices
        index = random.sample(list(torch.arange(instances_number)), K + Q)
        # initialize the spectrogram tensor
        spectrogram = torch.empty([0, 128, 51])
        for i in index:
            # concatenate spectrogram_buf with spectrogram to get a new tensor 
            # of random sampled instances of the word
            spectrogram = torch.cat((spectrogram, (spectrogram_buf[i, :, :]).view(1, y_dim, z_dim)), axis = 0)
        # concatenate the first K spectrograms with the support set
        support =  torch.cat((support, (spectrogram[:K]).view(1, K, y_dim, z_dim)), axis = 0)
        # concatenate the last Q spectrograms with the query set
        query = torch.cat((query, (spectrogram[K:K+Q]).view(1, Q, y_dim, z_dim)), axis = 0)
    return query, support

def get_training_validation_readers(features_folder, C):
  """
  get_training_validation_readers returns training and validation readers from the . 
  From the training and validation readers, it takes only the readers with at least C words
  and split the list in training readers and validation readers.

  Parameters:
  features_folder (list of string): list of the path of the training and validation
  C (int): number of classes

  Returns:
  training_readers (list of string): list of the training readers paths
  validation_readers (list of string): list of the validation readers paths
  """
  readers_path = []
  train_val_readers = []
  # scan each reader folder in the feature_folder
  for entry in os.scandir(features_folder):
      # create a list of reader names
      readers_path.append(entry.path)
  for reader_path in readers_path:
    words = []
    for word in os.scandir(reader_path):
            # create a list containing the path of the words
            words.append(word.path)
    if (len(words) >= C):
      train_val_readers.append(reader_path)
  train_val_readers = random.sample(train_val_readers, len(train_val_readers))
  training_readers = train_val_readers[:int(138/153*len(train_val_readers))]
  validation_readers = train_val_readers[int(138/153*len(train_val_readers)):]
  return training_readers, validation_readers


if torch.cuda.is_available():
  device = torch.device("cuda")
  print("Device: {}".format(device))
  print("Device name: {}".format(torch.cuda.get_device_properties(device).name))
else:
  device = torch.device("cpu")

C = 2 # classes
K = 1 # instances per class
Q = 16 # query set size

# To construct a C-way K-shot training episode, we randomly sample a reader from the training set, 
# sample C word classes from the reader, and sample K instances per class as the support set.

train_loss = []
train_acc = []

model = Protonet()
if torch.cuda.is_available():
  model.to(device='cuda')

print("Model parameters: {}".format(count_parameters(model)))

optim = torch.optim.Adam(model.parameters(), lr = 0.001)

training_readers, validation_readers = get_training_validation_readers("Training_validation_features/", C)

for episode in trange(60000, desc = "episode", position = 0, leave = True):
    query, support = batch_sample(training_readers, C, K, Q)
    if torch.cuda.is_available():
      support = support.to(device='cuda')
      query = query.to(device='cuda')
    
    model.train()
    optim.zero_grad()
    loss_out, acc_val = loss(support, query, model, "euclidean_dist")
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