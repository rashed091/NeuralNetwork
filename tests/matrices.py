import numpy as np
import torch

arr = [[1, 2], [3, 4]]

print(torch.Tensor(arr))

torch.manual_seed(0)

print(torch.rand(2, 2))

a = torch.ones(2, 2)

print(a.size())

print(a.view(-1).size())

print(a.view(4))

b = torch.ones(2, 2)

print(a.add_(b))


d = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
print(d.mean(dim=1))
