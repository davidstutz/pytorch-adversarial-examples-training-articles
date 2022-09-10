import torch
import common.torch
import common.numpy
import common.summary
from common.progress import ProgressBar


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
