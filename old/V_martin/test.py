import torch
from torch.nn.functional import mse_loss

a = torch.rand((3,))
b = torch.rand((3,))
l = [a, b]

print(torch.cat(l))