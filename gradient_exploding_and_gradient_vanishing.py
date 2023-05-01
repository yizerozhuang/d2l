import torch

if __name__ == '__main__':
    x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
    y = torch.sigmoid(x)
    y.backward(torch.ones_like(x))
    print(x)
    print(y)
    M = torch.normal(0, 1, size=(4, 4))
    print(M)
    for i in range(100):
        M = torch.mm(M, torch.normal(0, 1, size=(4, 4)))
    print(M)