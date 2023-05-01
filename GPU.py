import torch
from torch import nn

if __name__ == '__main__':
    print(torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1'))
    print(torch.cuda.device_count())