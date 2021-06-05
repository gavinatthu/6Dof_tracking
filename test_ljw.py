import torch

a = torch.randn(4,3)
print(a)
a = a.unsqueeze(2)
a = a.repeat(1,1,10)
print(a[:,:,0])