import torch
from torch import nn
from torch.nn import functional as F


def apply_init(self, inputs, init=None):
    self.forward(*inputs)
    if init is not None:
        self.net.apply(init)


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add_module(f"block{i}", block1())
    return net


def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)


def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)


def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)


def my_init(m):
    if type(m) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in m.named_parameters()][0])
        nn.init.uniform_(m.weight, -10, 10)
        m.weight.data *= m.weight.data.abs() >= 5


class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))


class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=True)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()


class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        return X - X.mean()


class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units, ))

    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))


if __name__ == '__main__':
    net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

    X = torch.rand(2, 20)
    print(net(X))

    net = MLP()
    print(net(X))

    net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
    print(net(X))

    net = FixedHiddenMLP()
    print(net(X))

    chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
    print(chimera(X))

    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    X = torch.rand(size=(2, 4))
    print(net(X))
    print(net[2].state_dict())
    print(type(net[2].bias))
    print(net[2].bias)
    print(net[2].bias.data)

    print(net[2].weight is None)
    print(*[(name, param.shape) for name, param in net[0].named_parameters()])
    print(*[(name, param.shape) for name, param in net.named_parameters()])
    print(net.state_dict()["2.bias"].data)
    rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
    print(rgnet(X))
    print(rgnet)
    print(rgnet[0][1][0].bias.data)

    net.apply(init_normal)
    print(net[0].weight.data[0], net[0].bias.data[0])

    net.apply(init_constant)
    print(net[0].weight.data[0], net[0].bias.data[0])

    net[0].apply(init_xavier)
    net[2].apply(init_42)

    print(net[0].weight.data[0])
    print(net[2].weight.data)

    net.apply(my_init)
    print(net[0].weight[:2])

    shared = nn.Linear(8, 8)
    net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Linear(8, 1))
    net(X)
    print(net[2].weight.data[0] == net[4].weight.data[0])
    net[2].weight.data[0, 0] = 100
    print(net[2].weight.data[0] == net[4].weight.data[0])

    net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
    print(net[0].weight)
    print(net)
    X = torch.rand(2, 20)
    net(X)
    print(net)

    layer = CenteredLayer()
    print(layer(torch.FloatTensor([1, 2, 3, 4, 5])))

    net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
    Y = net(torch.rand(4, 8))
    print(Y.mean())

    linear = MyLinear(5, 3)
    print(linear.weight)

    print(linear(torch.rand(2, 5)))

    net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
    print(net(torch.rand(2, 64)))

    x = torch.arange(4)
    torch.save(x, "x-file")

    x2 = torch.load("x-file")
    print(x2)

    y = torch.zeros(4)
    torch.save([x, y], "x-files")
    x2, y2 = torch.load("x-files")
    print(x2, y2)
    mydict = {"x": x, "y": y}
    torch.save(mydict, "mydict")
    mydict2 = torch.load("mydict")
    print(mydict2)

    net = MLP()
    X = torch.randn(size=(2, 20))
    Y = net(X)

    torch.save(net.state_dict(), "mlp.params")

    clone = MLP()
    clone.load_state_dict(torch.load("mlp.params"))
    print(clone.eval())

    Y_clone = clone(X)
    print(Y_clone==Y)