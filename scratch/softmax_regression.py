import torch
from util.accumulator import Accumulator
from scratch.linear_regression import sgd
from util.data import load_data_fashion_mnist
from util.net import softmax_net
from loss import cross_entropy
from util.visualization import show_images
from util.animator import Animator

def get_fashion_mnist_labels(labels):
    text_labels = ["t-shirt", "trouser", "pullover", "dress", "coat",
                   "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return [text_labels[int(i)] for i in labels]


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(param, net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                metric.add(accuracy(net(X), y), y.numel())
            else:
                metric.add(accuracy(net(param, X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(net, train_iter, loss, updater, param=None, lr=None):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        if isinstance(net, torch.nn.Module):
            y_hat = net(X)
        else:
            y_hat = net(param, X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(param, lr, X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater, param=None, lr=None):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=["train loss", "train acc", "test acc"])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater, param, lr)
        test_acc = evaluate_accuracy(param, net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
        print(
            f"epoch {epoch + 1:d}, loss {train_metrics[0]:.4f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}")
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc



def predict_ch3(net, test_iter, param=None, n=6):
    X, y = next(iter(test_iter))
    trues = get_fashion_mnist_labels(y)
    if isinstance(net, torch.nn.Module):
        preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    else:
        preds = get_fashion_mnist_labels(net(param, X).argmax(axis=1))
    acc = sum([1 if true == pred else 0 for true, pred in zip(trues, preds)]) / len(trues) * 100
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n], scale=2.5)
    print(f"the Accuracy is: {acc:.2f} %")


if __name__ == "__main__":
    batch_size = 256
    num_worker = 4
    train_iter, test_iter = load_data_fashion_mnist(batch_size, num_worker)
    X, y = next(iter(load_data_fashion_mnist(18,)[0]))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    num_inputs = 784
    num_outputs = 10
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)
    lr = 0.1
    num_epochs = 10
    net = softmax_net
    train_ch3(net=net, train_iter=train_iter, test_iter=test_iter, \
              loss=cross_entropy, num_epochs=num_epochs, updater=sgd, param=[W, b], lr=lr)
    predict_ch3(net, test_iter, [W, b])
