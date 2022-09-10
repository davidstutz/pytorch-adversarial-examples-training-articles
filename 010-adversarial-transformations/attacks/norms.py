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


class L2Norm(Norm):
    def __call__(self, perturbations):
        """
        Norm.

        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        return torch.norm(perturbations.view(perturbations.size()[0], -1), p=2, dim=1)

    def normalize(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        if len(gradients.size()) == 4:
            gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), 2, 1).view(-1, 1, 1, 1) + 1e-10)
        elif len(gradients.size()) == 2:
            gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), 2, 1).view(-1, 1) + 1e-10)
        else:
            assert False

    def scale(self, gradients):
        """
        Normalization.

        :param gradients: gradients
        :type gradients: torch.autograd.Variable
        """

        if len(gradients.size()) == 4:
            gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), 2, 1).view(-1, 1, 1, 1) + 1e-10)
        elif len(gradients.size()) == 2:
            gradients.data = torch.div(gradients.data, torch.norm(gradients.data.view(gradients.size()[0], -1), 2, 1).view(-1, 1) + 1e-10)
        else:
            assert False


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

        if len(gradients.size()) == 4:
            gradients.data = torch.div(gradients.data, torch.max(torch.abs(gradients.data.view(gradients.size()[0], -1)), dim=1)[0].view(-1, 1, 1, 1))
        elif len(gradients.size()) == 2:
            gradients.data = torch.div(gradients.data, torch.max(torch.abs(gradients.data.view(gradients.size()[0], -1)), dim=1)[0].view(-1, 1))
        else:
            assert False

