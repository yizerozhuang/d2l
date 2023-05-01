import torch
from torch import nn
from scratch.softmax_regression import train_ch3, load_data_fashion_mnist, predict_ch3
from util.net import init_weights
from util.visualization import show_images
from scratch.softmax_regression import get_fashion_mnist_labels


if __name__ == "__main__":
    X, y = next(iter(load_data_fashion_mnist(18,)[0]))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10
    train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, loss=loss, num_epochs=num_epochs, updater=trainer)
    predict_ch3(net, test_iter)
