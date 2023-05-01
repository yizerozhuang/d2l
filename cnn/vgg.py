import torch
from torch import nn
from util.data import load_data_fashion_mnist
from alex_net import train_ch6

def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(
        *conv_blks,
        nn.Flatten(),
        nn.Linear(out_channels *7 * 7, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 10)
    )

if __name__ == '__main__':
    conv_arch = ((1, 64), (1,128), (2,256), (2,512), (2,512))
    net = vgg(conv_arch)
    X = torch.randn(size=(1, 1, 244, 244))
    for blk in net:
        X = blk(X)
        print(blk.__class__.__name__, "output shape:\t", X.shape)
    lr, num_epochs, batch_size = 0.1, 10, 128
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
    train_ch6(net, train_iter, test_iter, num_epochs, lr, torch.device("cpu"))
