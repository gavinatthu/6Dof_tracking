import torch,time,os
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


cfg_fc = [800,10]

thresh = 0.4
lens = 0.5
decay = 0.25
num_classes = 10
batch_size  = 200
num_epochs = 50
learning_rate = 1e-3
input_dim = 2312
time_window = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float()


probs = 0.4
act_fun = ActFun.apply

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch>1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


class SNN_Model(nn.Module):

    def __init__(self, num_classes=3):
        super(SNN_Model, self).__init__()

        # self.conv1 = nn.Conv2d(2, cfg_cnn[0][1], kernel_size=3, stride=1, padding=1, )
        #
        # in_planes, out_planes, stride = cfg_cnn[1]
        # self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )
        #
        # in_planes, out_planes, stride = cfg_cnn[2]
        # self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, )


        self.fc1 = nn.Linear(input_dim , cfg_fc[0], )
        self.fc2 = nn.Linear(cfg_fc[0], cfg_fc[1], )
        #self.fc3 = nn.Linear(cfg_fc[1], cfg_fc[2], )

        # self.fc3.weight.data = self.fc3.weight.data * 0.1

        self.alpha1 = torch.nn.Parameter((1e-2 * torch.ones(1)).cuda(), requires_grad=True)
        self.alpha2 = torch.nn.Parameter((1e-2 * torch.ones(1)).cuda(), requires_grad=True)

        self.eta1 = torch.nn.Parameter((1e-1 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)
        self.eta2 = torch.nn.Parameter((1e-1 * torch.rand(1, cfg_fc[1])).cuda(), requires_grad=True)

        self.gamma1 = torch.nn.Parameter((1e-2 * torch.rand(cfg_fc[0], cfg_fc[0])).cuda(), requires_grad=True)
        self.gamma2 = torch.nn.Parameter((1e-2 * torch.rand(cfg_fc[1], cfg_fc[1])).cuda(), requires_grad=True)

        self.beta1 = torch.nn.Parameter((1e-2 * torch.rand(1, input_dim)).cuda(), requires_grad=True)
        self.beta2 = torch.nn.Parameter((1e-2 * torch.rand(1, cfg_fc[0])).cuda(), requires_grad=True)

    def produce_hebb(self):
        hebb1 = torch.zeros(input_dim, cfg_fc[0], device=device)
        hebb2 = torch.zeros(cfg_fc[0], cfg_fc[1], device=device)
        return (hebb1, hebb2)


    def forward(self, input, hebb, win = time_window):


        h1_mem = h1_spike = h1_sumspike = torch.zeros(batch_size, cfg_fc[0], device=device)
        h2_mem = h2_spike = h2_sumspike = torch.zeros(batch_size, cfg_fc[1], device=device)

        hebb1, hebb2  = hebb

        for step in range(win):
            k_filter = 1

            x = input[:, :, :, :, step]

            x = x.view(batch_size, -1)

            h1_mem, h1_spike, hebb1 = mem_update(self.fc1, self.alpha1, self.beta1, self.gamma1, self.eta1, x, h1_spike,
                                                 h1_mem, hebb1)

            h2_mem, h2_spike, hebb2 = mem_update(self.fc2, self.alpha2, self.beta2, self.gamma2, self.eta2, h1_spike,
                                                 h2_spike, h2_mem, hebb2)

            # h2_sumspike  = h2_sumspike + h2_spike

        outs = h2_mem / thresh
        return outs, (hebb1.data, hebb2.data)




def mem_update(fc, alpha, beta, gamma,  eta, inputs, spike, mem, hebb):
    state = fc(inputs) + alpha * inputs.mm(hebb)
    mem = mem * (1 - spike) * decay + state
    now_spike = act_fun(mem - thresh)
    hebb = 0.99 * hebb - torch.bmm((inputs * beta.clamp(min=0.)).unsqueeze(2),
                                   ((mem / thresh) - eta).unsqueeze(1)).mean(dim=0).squeeze()
    hebb = hebb.clamp(min=-5, max=5)
    return mem, now_spike.float(), hebb


def mem_update_no_plastic(fc, inputs, spike, mem):
    state = fc(inputs)
    mem = mem * (1 - spike) * decay + state
    now_spike = act_fun(mem - thresh)

    return mem, now_spike.float()


