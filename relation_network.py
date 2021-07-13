import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    """
    CNNEncoder is the embedding module. The architecture consists of 4 convolutional block contains a 
    64-filter 3 X 3 convolution, a batch normalisation and a ReLU nonlinearity layer respectively. 
    The first 3 blocks also contain a 2 X 2 max-pooling layer while the last two do not. We do so 
    because we need the output feature maps for further convolutional layers in the relation module
    """
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
                        #nn.MaxPool2d(2))

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = out.view(out.size(0),-1)
        return out # 64

class RelationNetwork(nn.Module):
    """
    The RelationNetwork is the relation module. It consists of two convolutional blocks and
    two fully-connected layers. Each of convolutional block is a 3 X 3 convolution with 64 filters 
    followed by batch normalisation, ReLU non-linearity and 2 X 2 max-pooling.
    The two fully-connected layers are 8 and 1 dimensional, respectively. All fully-connected layers are
    ReLU except the output layer is Sigmoid in order to generate relation scores in a reasonable range 
    for all versions of our network architecture.
    """
    def __init__(self,input_size,hidden_size):
        super(RelationNetwork, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(128,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        out = self.layer1(x)    #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X FEATURE_DIM X 2 X 2
        out = self.layer2(out)  #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X FEATURE_DIM X 1 X 1
        out = out.view(out.size(0),-1)   #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X FEATURE_DIM
        out = F.relu(self.fc1(out))      #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X RELATION_DIM
        #out = F.sigmoid(self.fc2(out)) # deprecated
        out = torch.sigmoid(self.fc2(out))  #  (CLASS_NUM * BATCH_NUM_PER_CLASS * 5) X 1
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        m.bias.data = torch.ones(m.bias.data.size())