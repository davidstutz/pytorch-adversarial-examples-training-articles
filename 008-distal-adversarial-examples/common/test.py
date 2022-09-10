import torch
import numpy
import common.torch
import common.numpy
import common.summary
from common.progress import ProgressBar
from common.log import log


def test(model, testset, eval=True, loss=True, cuda=False):
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
    probabilities = None

    # should work with and without labels
    for b, data in enumerate(testset):
        targets = None
        if isinstance(data, tuple) or isinstance(data, list):
            inputs = data[0]
            targets = data[1]
        else:
            inputs = data

        assert isinstance(inputs, torch.Tensor)

        inputs = common.torch.as_variable(inputs, cuda)
        targets = common.torch.as_variable(targets, cuda)

        logits = model(inputs)
        check_nan = torch.sum(logits, dim=1)
        check_nan = (check_nan != check_nan)
        logits[check_nan, :] = 0.1
        if torch.any(check_nan):
            log('corrected %d nan rows' % torch.sum(check_nan))

        probabilities_ = common.torch.softmax(logits, dim=1).detach().cpu().numpy()
        probabilities = common.numpy.concatenate(probabilities, probabilities_)

        if targets is not None and loss:
            targets = common.torch.as_variable(targets, cuda)
            error = common.torch.classification_error(logits, targets)
            loss = common.torch.classification_loss(logits, targets)
            progress('test', b, len(testset), info='error=%g loss=%g' % (error.item(), loss.item()))
        else:
            progress('test', b, len(testset))

    #assert probabilities.shape[0] == len(testset.dataset)

    return probabilities


def attack(model, testset, attack, objective, attempts=1, writer=common.summary.SummaryWriter(), cuda=False):
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
    :param attempts: number of attempts
    :type attempts: int
    :param get_writer: summary writer or utility function to get writer
    :type get_writer: torch.utils.tensorboard.SummaryWriter or callable
    :param cuda: whether to use CUDA
    :type cuda: bool
    """

    assert model.training is False
    assert len(testset) > 0
    assert attempts >= 0
    assert isinstance(testset, torch.utils.data.DataLoader)
    assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
    assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

    perturbations = []
    probabilities = []
    errors = []

    # should work via subsets of datasets
    for a in range(attempts):
        perturbations_a = None
        probabilities_a = None
        errors_a = None

        for b, data in enumerate(testset):
            assert isinstance(data, tuple) or isinstance(data, list)

            inputs = common.torch.as_variable(data[0], cuda)
            labels = common.torch.as_variable(data[1], cuda)

            # attack target labels
            targets = None
            if len(list(data)) > 2:
                targets = common.torch.as_variable(data[2], cuda)

            objective.set(labels, targets)
            attack.progress = ProgressBar()
            perturbations_b, errors_b = attack.run(model, inputs, objective,
                                                   writer=writer if not callable(writer) else writer('%d-%d' % (a, b)),
                                                   prefix='%d/%d/' % (a, b) if not callable(writer) else '')

            inputs = inputs + common.torch.as_variable(perturbations_b, cuda)
            logits = model(inputs)
            probabilities_b = common.torch.softmax(logits, dim=1).detach().cpu().numpy()

            perturbations_a = common.numpy.concatenate(perturbations_a, perturbations_b)
            probabilities_a = common.numpy.concatenate(probabilities_a, probabilities_b)
            errors_a = common.numpy.concatenate(errors_a, errors_b)

        perturbations.append(perturbations_a)
        probabilities.append(probabilities_a)
        errors.append(errors_a)

    perturbations = numpy.array(perturbations)
    probabilities = numpy.array(probabilities)
    errors = numpy.array(errors)

    assert perturbations.shape[1] == len(testset.dataset)
    assert probabilities.shape[1] == len(testset.dataset)
    assert errors.shape[1] == len(testset.dataset)

    return perturbations, probabilities, errors
    # attempts x N x C x H x W, attempts x N x K, attempts x N