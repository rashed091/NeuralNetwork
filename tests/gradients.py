import torch
import numpy as np


a = torch.ones((2, 2), requires_grad=True)

print(a.requires_grad)


x = torch.ones(2, requires_grad=True)
y = 5 * (x + 1) ** 2
print(y)

o = 0.5 * torch.sum(y)
print(o)

o.backward()

print(x.grad)
