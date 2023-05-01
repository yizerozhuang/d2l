import random
import torch
from util.data import synthetic_data
from util.visualization import set_figsize, scatter_plot
from util.net import linreg
from loss import squared_loss
from optimizer import sgd


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = torch.asarray(indices[i: min(i + batch_size, num_examples)])
        yield features[j], labels.take(j)

if __name__ == "__main__":
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    set_figsize()
    scatter_plot(features[:, 1], labels)

    batch_size = 10

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    lr = 0.03
    num_epochs = 3
    net = linreg
    loss = squared_loss

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

    print(true_w, w)
    print(true_b, b)
