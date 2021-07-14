import torch


def compute_loss(input_: torch.Tensor, target: torch.Tensor, time_start_idx: int=None):
    """
    :param input: tensor of shape (batch, 3, time)
    :param target: tensor of shape (batch, 3, time)
    :param time_start_idx: Time-index from which to start computing the loss
    :return: loss
    """
    assert len(input_.shape) == 3
    assert len(target.shape) == 3
    assert input_.shape == target.shape
    assert input_.shape[1] == 3

    if time_start_idx:
        input_ = input_[..., time_start_idx:]
        target = target[..., time_start_idx:]

    return torch.mean(torch.sqrt(torch.sum((input_ - target) ** 2, dim=1)))

def compute_loss_snn(input_: torch.Tensor, target: torch.Tensor):
    """
    :param input: tensor of shape (batch, 3)
    :param target: tensor of shape (batch, 3, time)
    :return: loss
    """
    assert len(input_.shape) == 2
    assert len(target.shape) == 3
    assert input_.shape[1] == 3
    target = torch.mean(target, dim = 2)
    return torch.mean(torch.sqrt(torch.sum((input_ - target) ** 2, dim=1)))


def compute_loss_snn_6Dof(input_: torch.Tensor, target: torch.Tensor):
    """
    :param input: tensor of shape (batch, 7)
    :param target: tensor of shape (batch, 7, time)
    :return: loss
    """
    assert len(input_.shape) == 2
    assert len(target.shape) == 3
    assert input_.shape[1] == 7
    target = torch.mean(target, dim = 2)
    loss_p = torch.mean(torch.sqrt(torch.sum((input_[:,:4] - target[:,:4])**2, dim=1)/torch.norm(target[:,:4], dim=1)))
    loss_o = torch.mean(torch.sqrt(torch.sum((input_[:,4:] - target[:,4:])**2, dim=1)/torch.norm(target[:,4:], dim=1)))
    return loss_p + loss_o
