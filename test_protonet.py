import torch
import os
import numpy
import random

p=5
n=10

# initialize support tensor of dimension 0 x K x 128 x 51
positive_set =  torch.empty([p, 128, 51])
negative_set =  torch.empty([n, 128, 51])
# initialize query tensor of dimension 0 x Q x 128 x 51
query_set = torch.empty([0, 128, 51])

words=[]
for audio in os.scandir("Test_features"):
    for word in os.scandir(audio):
        words.append(word)

    pos_word=random.sample(words,1)

    specs=torch.load("Test_features/"+pos_word)
    index=torch.arange(len(specs))
    pos_index=random.sample(index,p)
    for i in index:
        pos=specs(i)
        torch.cat(positive_set,pos)
    for i in index:
        if i not in pos_index:
            query=specs(i)
            torch.cat(query_set,query)

    words.remove(pos_word)

    for i in range(n):
        neg=random.sample(words,1)
        specs=torch.load("Test_features/"+neg)
        neg = random.sample(specs,1)
        torch.cat(negative_set,neg)



