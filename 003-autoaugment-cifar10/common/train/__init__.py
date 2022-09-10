import torch
import math
from .normal_training import *


def get_exponential_scheduler(optimizer, batches_per_epoch, gamma=0.97):
    """
    Get exponential scheduler.

    Note that the resulting optimizer's step function is called after each batch!

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param batches_per_epoch: number of batches per epoch
    :type batches_per_epoch: int
    :param gamma: gamma
    :type gamma: float
    :return: scheduler
    :rtype: torch.optim.lr_scheduler.LRScheduler
    """

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda epoch: gamma ** math.floor(epoch/batches_per_epoch)])


def get_multi_step_scheduler(optimizer, batches_per_epoch, milestones=[100, 150, 200], gamma=0.1):
    """
    Get step scheduler.

    Note that the resulting optimizer's step function is called after each batch!

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param batches_per_epoch: number of batches per epoch
    :type batches_per_epoch: int
    :param gamma: gamma
    :type gamma: float
    :return: scheduler
    :rtype: torch.optim.lr_scheduler.LRScheduler
    """

    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[milestone*batches_per_epoch for milestone in milestones], gamma=gamma)