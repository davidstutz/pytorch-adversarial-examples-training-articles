import torch
import math


class Norm:
    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        raise NotImplementedError()

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        raise NotImplementedError()

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        raise NotImplementedError()


class LInfNorm(Norm):
    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        return torch.max(torch.abs(perturbations.view(perturbations.size()[0], -1)), dim=1)[0]

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.sign(gradients.data)

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        gradients.data = torch.div(gradients.data, torch.max(torch.abs(gradients.data.view(gradients.size()[0], -1)), dim=1)[0].view(-1, 1, 1, 1))