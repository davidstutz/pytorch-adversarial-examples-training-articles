import torch
import common.torch
import common.numpy
import common.summary
from common.progress import ProgressBar


def test(model, testset, eval=True, cuda=False):
    """
    Test a model on a clean or adversarial dataset.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param cuda: use CUDA
    :type cuda: bool
    """

    assert model.training is not eval
    assert len(testset) > 0
    #assert isinstance(testset, torch.utils.data.DataLoader)
    #assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    overall_error = 0

    for b, (inputs, targets) in enumerate(testset):
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)

        logits = model(inputs)
        error = common.torch.classification_error(logits, targets)
        loss = common.torch.classification_loss(logits, targets)
        progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))

        overall_error += error
    overall_error /= len(testset)
    return overall_error


def attack(model, testset, attack, objective, writer=common.summary.SummaryWriter(), cuda=False):
    """
    Attack model.

    :param model: model
    :type model: torch.nn.Module
    :param testset: test set
    :type testset: torch.utils.data.DataLoader
    :param attack: attack
    :type attack: attacks.Attack
    :param objective: attack objective
    :type objective: attacks.Objective
    :param get_writer: summary writer or utility function to get writer
    :type get_writer: torch.utils.tensorboard.SummaryWriter or callable
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    progress = ProgressBar()
    overall_error = 0
    probabilities = None

    for b, (inputs, targets) in enumerate(testset):
        assert isinstance(inputs, torch.Tensor)
        assert isinstance(targets, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)

        # could be used for a targeted attack as well
        objective.set(targets)
        attack.progress = ProgressBar()
        perturbations_b, errors_b = attack.run(model, inputs, objective, writer=writer, prefix='')

        inputs = inputs + common.torch.as_variable(perturbations_b, cuda)
        logits = model(inputs)

        probabilities = common.numpy.concatenate(probabilities, torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy())
        error = common.torch.classification_error(logits, targets)
        loss = common.torch.classification_loss(logits, targets)

        progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))

        overall_error += error
    overall_error /= len(testset)
    return overall_error, probabilities