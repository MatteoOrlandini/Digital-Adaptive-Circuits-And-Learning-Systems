import torch.nn as nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Protonet(nn.Module):
    def __init__(self):
        super(Protonet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,64),
            conv_block(64,64),
            conv_block(64,64),
            conv_block(64,64)
        )
    def forward(self,x):
        (num_samples,mel_bins, seq_len) = x.shape
        #print("x.shape:",x.shape)
        x = x.view(-1,1,mel_bins,seq_len)
        #print("x.view:",x.shape)
        x = self.encoder(x)
        #print("x.encoder:",x.shape)
        return x.view(x.size(0),-1)