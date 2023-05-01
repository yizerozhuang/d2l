import math
import numpy as np
import torch
from torch import nn
from util.accumulator import Accumulator
from matplotlib import pyplot as plt
from scratch.linear_regression import synthetic_data, linreg, squared_loss, sgd
from scratch.softmax_regression import train_epoch_ch3
from d2l import torch as d2l


def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None, legend=None,
             figsize=(3.5, 2.5)):
    plt.figure(figsize=figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals)
        plt.legend(legend)
    plt.show()


def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]


def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    for epoch in range(num_epochs):
        train_epoch_ch3(net, train_iter, loss, trainer)
        # if epoch == 0 or (epoch + 1) % 20 ==0:
        #     semilogy(epoch + 1, (evaluate_loss(net, train_iter, loss),  epoch + 1, (evaluate_loss(net, test_iter, loss))))
    print("weight: ", net[0].weight.data.numpy())


def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train_regularization(lambd):
    w, b = init_params()
    net, loss = lambda X: linreg(X, w, b), squared_loss
    num_epochs, lr = 100, 0.003
    for epoch in range(num_epochs):
        for X, y in train_iter:
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:d}, avg_loss {float((sum(l)/len(l)).detach()):.4f}, train loss {evaluate_loss(net, train_iter, loss):.3f}, test loss {evaluate_loss(net, test_iter, loss):.3f}")
    print("The L2 norm for w is ", torch.norm(w).item())

def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()
    loss = nn.MSELoss(reduction="none")
    num_epochs, lr = 100, 0.003
    trainer = torch.optim.SGD([
        {"params": net[0].weight, "weight_decay": wd},
        {"params": net[0].bias}
    ], lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            trainer.step()
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch + 1:d}, avg_loss {float((sum(l)/len(l)).detach()):.4f}, train loss {evaluate_loss(net, train_iter, loss):.3f}, test loss {evaluate_loss(net, test_iter, loss):.3f}")
    print("The L2 norm for w is ", net[0].weight.norm().item())

if __name__ == '__main__':
    max_degree = 20
    n_train, n_test = 100, 100
    true_w = np.zeros(max_degree)
    true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

    features = np.random.normal(size=(n_train + n_test, 1))
    np.random.shuffle(features)
    poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
    for i in range(max_degree):
        poly_features[:, i] /= math.gamma(i + 1)
    labels = np.dot(poly_features, true_w)
    labels += np.random.normal(scale=0.1, size=labels.shape)

    true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) for x in
                                               [true_w, features, poly_features, labels]]

    print(features[:2], poly_features[:2, :], labels[:2])
    train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])
    train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])
    train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:])

    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)
    train_regularization(0)
    train_regularization(3)
    train_concise(0)
    train_concise(3)