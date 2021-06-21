import torch

x = torch.tensor([0, 1, 2, 3, 4])
torch.save(x, './pretrained/tensor.pt')