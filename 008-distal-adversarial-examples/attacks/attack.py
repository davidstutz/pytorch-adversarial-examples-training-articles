import common.summary
import torch
from .objectives import *


class Attack:
    """
    Generic attack.
    """

    def __init__(self):
        """
        Constructor.
        """

        self.progress = None
        """ (common.progress.ProgressBar) Progress bar. """

    # objective is not in constructor as it keeps knowledge about targets/true labels
    def run(self, model, images, objective, writer=common.summary.SummaryWriter(), prefix=''):
        """
        Run attack.

        :param model: model to attack
        :type model: torch.nn.Module
        :param images: images
        :type images: torch.autograd.Variable
        :param objective: objective
        :type objective: UntargetedObjective or TargetedObjective
        :param writer: summary writer
        :type writer: common.summary.SummaryWriter
        :param prefix: prefix for writer
        :type prefix: str
        """

        assert model.training is False
        assert isinstance(images, torch.autograd.Variable)
        assert isinstance(objective, Objective)
        assert common.torch.is_cuda(model) == common.torch.is_cuda(images)

        writer.add_text('%sobjective' % prefix, objective.__class__.__name__)

    def summary_dict(self, values):
        """
        Helper for summaries, converts array of values to dict;

        :param values: values
        :type values: numpy.ndarray or torch.Tensor
        :return: values in dict
        :rtype: dict
        """

        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        values = values.tolist()
        keys = list(map(str, range(len(values))))

        return dict(zip(keys, values))