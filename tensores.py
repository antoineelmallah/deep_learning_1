import torch

x = torch.tensor([[1,2,3],[4,5,6]], dtype=float, requires_grad=True)
print(x)
print(x.shape)

print()

x = x.view(-1)
print(x)
print(x.shape)

print()

y = torch.tensor(range(len(x)))
print(y)

print()

z = x + y
print(z)