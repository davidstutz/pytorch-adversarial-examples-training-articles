import torch
import numpy
import math
import scipy
from copy import deepcopy
import functools
import common.numpy as cnumpy
import builtins


def is_cuda(mixed):
    """
    Check if model/tensor is on CUDA.

    :param mixed: model or tensor
    :type mixed: torch.nn.Module or torch.autograd.Variable or torch.Tensor
    :return: on cuda
    :rtype: bool
    """

    assert isinstance(mixed, torch.nn.Module) or isinstance(mixed, torch.autograd.Variable) \
        or isinstance(mixed, torch.Tensor), 'mixed has to be torch.nn.Module, torch.autograd.Variable or torch.Tensor'

    is_cuda = False
    if isinstance(mixed, torch.nn.Module):
        is_cuda = True
        for parameters in list(mixed.parameters()):
            is_cuda = is_cuda and parameters.is_cuda
    if isinstance(mixed, torch.autograd.Variable):
        is_cuda = mixed.is_cuda
    if isinstance(mixed, torch.Tensor):
        is_cuda = mixed.is_cuda

    return is_cuda


def as_variable(mixed, cuda=False, requires_grad=False):
    """
    Get a tensor or numpy array as variable.

    :param mixed: input tensor
    :type mixed: torch.Tensor or numpy.ndarray
    :param device: gpu or not
    :type device: bool
    :param requires_grad: gradients
    :type requires_grad: bool
    :return: variable
    :rtype: torch.autograd.Variable
    """

    assert isinstance(mixed, numpy.ndarray) or isinstance(mixed, torch.Tensor), 'input needs to be numpy.ndarray or torch.Tensor'

    if isinstance(mixed, numpy.ndarray):
        mixed = torch.from_numpy(mixed)

    if cuda:
        mixed = mixed.cuda()
    return torch.autograd.Variable(mixed, requires_grad)


def concatenate(tensor1, tensor2, axis=0):
    """
    Basically a wrapper for torch.dat, with the exception
    that the array itself is returned if its None or evaluates to False.

    :param tensor1: input array or None
    :type tensor1: mixed
    :param tensor2: input array
    :type tensor2: numpy.ndarray
    :param axis: axis to concatenate
    :type axis: int
    :return: concatenated array
    :rtype: numpy.ndarray
    """

    assert isinstance(tensor2, torch.Tensor) or isinstance(tensor2, torch.autograd.Variable)
    if tensor1 is not None:
        assert isinstance(tensor1, torch.Tensor) or isinstance(tensor1, torch.autograd.Variable)
        return torch.cat((tensor1, tensor2), axis=axis)
    else:
        return tensor2


def classification_error(logits, targets, reduction='mean'):
    """
    Accuracy.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduce: reduce to number or keep per element
    :type reduce: bool
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        values, indices = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)
    else:
        indices = torch.round(torch.nn.functional.sigmoid(logits)).view(-1)

    errors = torch.clamp(torch.abs(indices.long() - targets.long()), max=1)
    if reduction == 'mean':
        return torch.mean(errors.float())
    elif reduction == 'sum':
        return torch.sum(errors.float())
    else:
        return errors


def softmax(logits, dim=1):
    """
    Softmax.

    :param logits: logits
    :type logits: torch.Tensor
    :param dim: dimension
    :type dim: int
    :return: softmax
    :rtype: torch.Tensor
    """

    if logits.size()[1] > 1:
        return torch.nn.functional.softmax(logits, dim=dim)
    else:
        probabilities = torch.nn.functional.sigmoid(logits)
        return torch.cat((1 - probabilities,  probabilities), dim=dim)


def classification_loss(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert logits.size()[0] == targets.size()[0]
    assert len(list(targets.size())) == 1# or (len(list(targets.size())) == 2 and targets.size(1) == 1)
    assert len(list(logits.size())) == 2

    if logits.size()[1] > 1:
        return torch.nn.functional.cross_entropy(logits, targets, reduction=reduction)
    else:
        # probability 1 is class 1
        # probability 0 is class 0
        return torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(logits).view(-1), targets.float(), reduction=reduction)


def project_ball(tensor, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param tensor: variable or tensor
    :type tensor: torch.autograd.Variable or torch.Tensor
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(tensor, torch.Tensor) or isinstance(tensor, torch.autograd.Variable), 'given tensor should be torch.Tensor or torch.autograd.Variable'

    if ord == 1:
        # ! Does not allow differentiation obviously!
        cuda = is_cuda(tensor)
        array = tensor.detach().cpu().numpy()
        array = cnumpy.project_ball(array, epsilon=epsilon, ord=ord)
        tensor = torch.from_numpy(array)
        if cuda:
            tensor = tensor.cuda()
    elif ord == 2:
        size = list(tensor.shape)
        flattened_size = int(numpy.prod(size[1:]))

        tensor = tensor.view(-1, flattened_size)
        clamped = torch.clamp(epsilon/torch.norm(tensor, 2, dim=1), max=1)
        clamped = clamped.view(-1, 1)

        tensor = tensor * clamped
        if len(size) == 4:
            tensor = tensor.view(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            tensor = tensor.view(-1, size[1])
    elif ord == float('inf'):
        tensor = torch.clamp(tensor, min=-epsilon, max=epsilon)
    else:
        raise NotImplementedError()

    return tensor


def expand_as(tensor, tensor_as):
    """
    Expands the tensor using view to allow broadcasting.

    :param tensor: input tensor
    :type tensor: torch.Tensor or torch.autograd.Variable
    :param tensor_as: reference tensor
    :type tensor_as: torch.Tensor or torch.autograd.Variable
    :return: tensor expanded with singelton dimensions as tensor_as
    :rtype: torch.Tensor or torch.autograd.Variable
    """

    view = list(tensor.size())
    for i in range(len(tensor.size()), len(tensor_as.size())):
        view.append(1)

    return tensor.view(view)


def max_p_loss(logits, targets=None, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    max_log = torch.max(torch.nn.functional.softmax(logits, dim=1), dim=1)[0]
    if reduction == 'mean':
        return torch.mean(max_log)
    elif reduction == 'sum':
        return torch.sum(max_log)
    else:
        return max_log


def max_log_loss(logits, targets=None, reduction='mean'):
    """
    Loss.

    :param logits: predicted classes
    :type logits: torch.autograd.Variable
    :param targets: target classes
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    max_log = torch.max(torch.nn.functional.log_softmax(logits, dim=1), dim=1)[0]
    if reduction == 'mean':
        return torch.mean(max_log)
    elif reduction == 'sum':
        return torch.sum(max_log)
    else:
        return max_log


class GaussianLayer(torch.nn.Module):
    """
    Gaussian convolution.

    See https://pytorch.org/docs/stable/nn.html.
    """

    def __init__(self, sigma=3, channels=3):
        """

        """
        super(GaussianLayer, self).__init__()

        self.sigma = sigma
        """ (float) Sigma. """

        padding = math.ceil(self.sigma)
        kernel = 2*padding + 1

        self.seq = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((padding, padding, padding, padding)),
            torch.nn.Conv2d(channels, channels, kernel, stride=1, padding=0, bias=None, groups=channels)
        )

        n = numpy.zeros((kernel, kernel))
        n[padding, padding] = 1

        k = scipy.ndimage.gaussian_filter(n, sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return self.seq(input)