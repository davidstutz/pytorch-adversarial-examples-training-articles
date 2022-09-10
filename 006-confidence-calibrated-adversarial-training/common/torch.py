import torch
import numpy
import common.numpy as cnumpy


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


def power_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Power transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    # returned value determines importance of uniform distribution:
    # (1 - ret)*one_hot + ret*uniform

    norms = norm(perturbations)
    return 1 - torch.pow(1 - torch.min(torch.ones_like(norms), norms / epsilon), gamma), norms


def tanh_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Linear transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    norms = norm(perturbations)
    return torch. tanh(1 / (epsilon / gamma) * norms)


def exponential_transition(perturbations, norm, epsilon=0.3, gamma=1):
    """
    Exponential transition rule.

    :param perturbations: perturbations
    :type perturbations: torch.autograd.Variable
    :param norm: norm
    :type norm: attacks.norms.Norm
    :param epsilon: epsilon
    :type epsilon: float
    :param gamma: gamma
    :type gamma: float
    :return: gamma, norms
    :rtype: torch.autograd.Variable, torch.autograd.Variable
    """

    norms = norm(perturbations)
    return 1 - torch.exp(-gamma * norms), norms


def cross_entropy_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = torch.sum(- targets * torch.nn.functional.log_softmax(logits, dim=1), dim=1)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def left_sided_kullback_leibler_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = - torch.mul(targets, torch.log(targets + 1e-6) - torch.nn.functional.log_softmax(logits, dim=1))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def right_sided_kullback_leibler_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    divergences = - torch.mul(torch.nn.functional.softmax(logits, dim=1), torch.nn.functional.logsoftmax(logits, dim=1) - torch.log(targets + 1e-6))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def jensen_shannon_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    return 0.5*left_sided_kullback_leibler_divergence(logits, targets) + 0.5*left_sided_kullback_leibler_divergence(logits, targets)


def bhattacharyya_coefficient(logits, targets):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :return: error
    :rtype: torch.autograd.Variable
    """

    assert len(list(logits.size())) == len(list(targets.size()))
    assert logits.size()[0] == targets.size()[0]
    assert logits.size()[1] == targets.size()[1]
    assert logits.size()[1] > 1

    # http://www.cse.yorku.ca/~kosta/CompVis_Notes/bhattacharyya.pdf
    # torch.sqrt not differentiable at zero
    return torch.clamp(torch.sum(torch.sqrt(torch.nn.functional.softmax(logits, dim=1) * targets + 1e-8), dim=1), min=0, max=1)


def bhattacharyya_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    divergences = - 2*torch.log(bhattacharyya_coefficient(logits, targets))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def hellinger_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    # torch.sqrt not differentiable at zero
    divergences = torch.sqrt(1 - bhattacharyya_coefficient(logits, targets) + 1e-8)
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def bhattacharyya_angle_divergence(logits, targets, reduction='mean'):
    """
    Loss.

    :param logits: predicted logits
    :type logits: torch.autograd.Variable
    :param targets: target distributions
    :type targets: torch.autograd.Variable
    :param reduction: reduction type
    :type reduction: str
    :return: error
    :rtype: torch.autograd.Variable
    """

    # torch.sqrt not differentiable at zero
    divergences = torch.acos(0.005 + 0.99*bhattacharyya_coefficient(logits, targets))
    if reduction == 'mean':
        return torch.mean(divergences)
    elif reduction == 'sum':
        return torch.sum(divergences)
    else:
        return divergences


def one_hot(classes, C):
    """
    Convert class labels to one-hot vectors.

    :param classes: classes as B x 1
    :type classes: torch.autograd.Variable or torch.Tensor
    :param C: number of classes
    :type C: int
    :return: one hot vector as B x C
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(classes, torch.autograd.Variable) or isinstance(classes, torch.Tensor), 'classes needs to be torch.autograd.Variable or torch.Tensor'
    assert len(classes.size()) == 2 or len(classes.size()) == 1, 'classes needs to have rank 2 or 1'
    assert C > 0

    if len(classes.size()) < 2:
        classes = classes.view(-1, 1)

    one_hot = torch.Tensor(classes.size(0), C)
    if is_cuda(classes):
         one_hot = one_hot.cuda()

    if isinstance(classes, torch.autograd.Variable):
        one_hot = torch.autograd.Variable(one_hot)

    one_hot.zero_()
    one_hot.scatter_(1, classes, 1)

    return one_hot