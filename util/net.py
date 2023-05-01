import torch
from torch import nn

def linreg(X, w, b):
    return torch.matmul(X, w) + b

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def softmax_net(param, X):
    return softmax(torch.matmul(X.reshape((-1, param[0].shape[0])), param[0]) + param[1])

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
