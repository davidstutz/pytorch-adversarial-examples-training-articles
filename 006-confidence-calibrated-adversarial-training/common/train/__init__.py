import torch
import math
from .normal_training import *
from .adversarial_training import *
from .confidence_calibrated_adversarial_training import *


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


def get_cyclic_scheduler(optimizer, batches_per_epoch, base_lr=10e-6, max_lr=0.1, step_size_factor=5):
    """
    Get cyclic scheduler.

    Note that the resulting optimizer's step function is called after each batch!

    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param batches_per_epoch: number of batches per epoch
    :type batches_per_epoch: int
    :param base_lr: base learning rate
    :type base_lr: float
    :param max_lr: max learning rate
    :type max_lr: float
    :param step_size_factor: step size in multiples of batches
    :type step_size_factor: int
    :return: scheduler
    :rtype: torch.optim.lr_scheduler.LRScheduler
    """

    return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=batches_per_epoch*step_size_factor)
