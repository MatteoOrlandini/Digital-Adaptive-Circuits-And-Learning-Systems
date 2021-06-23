import torch
import torch.nn.functional 
from torch.optim import Optimizer
from torch.nn import Module
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import torch.nn.functional as F

def proto_net_episode(model: Module,
                      optimiser: Optimizer,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      n_shot: int,
                      k_way: int,
                      q_queries: int,
                      train: bool):
    """Performs a single training episode for a Prototypical Network.
    # Arguments
        model: Prototypical Network to be trained.
        optimiser: Optimiser to calculate gradient step
        loss_fn: Loss function to calculate between predictions and outputs. Should be cross-entropy
        x: Input samples of few shot classification task
        y: Input labels of few shot classification task
        n_shot: Number of examples per class in the support set
        k_way: Number of classes in the few shot classification task
        q_queries: Number of examples per class in the query set
        distance: Distance metric to use when calculating distance between class prototypes and queries
        train: Whether (True) or not (False) to perform a parameter update
    # Returns
        loss: Loss of the Prototypical Network on this task
        y_pred: Predicted class probabilities for the query set on this task
    """

    if train:
        model.train()
        optimiser.zero_grad()
    else:
        model.eval()

    # Embed all samples
    embeddings = model(x)

    # Samples are ordered by the NShotWrapper class as follows:
    # k lots of n support samples from a particular class
    # k lots of q query samples from those classes
    support = embeddings[:n_shot*k_way]
    queries = embeddings[n_shot*k_way:]
    y_support = y[:n_shot * k_way]
    y_queries = y[n_shot * q_queries:]

    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    prototypes = support.reshape(k_way, n_shot, -1).mean(dim=1)

    # Calculate squared distances between all queries and all prototypes
    # Output should have shape (q_queries * k_way, k_way) = (num_queries, k_way)
    distances = (
            queries.unsqueeze(1).expand(queries.shape[0], support.shape[0], -1) -
            support.unsqueeze(0).expand(queries.shape[0], support.shape[0], -1)
    ).pow(2).sum(dim=2)


    # Calculate log p_{phi} (y = k | x)
    #log_p_y = (-distances).log_softmax(dim=1)
    log_p_y = F.log_softmax(-distances, dim=1).view(k_way, q_queries, -1)
    #loss = CrossEntropyLoss(log_p_y, y_queries)
    target_inds = torch.arange(0, k_way).view(k_way, 1, 1).expand(k_way, q_queries, 1).long()
    target_inds = Variable(target_inds, requires_grad=False)
    loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()

    # Prediction probabilities are softmax over distances
    y_pred = (-distances).softmax(dim=1)

    if train:
        # Take gradient step
        loss.backward()
        optimiser.step()

    return loss, y_pred