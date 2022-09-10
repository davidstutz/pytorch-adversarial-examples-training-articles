import common.torch
import torch
import numpy
import math
import scipy.ndimage
import random
import common.numpy


class Initialization:
    """
    Interface for initialization.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        raise NotImplementedError()


class SequentialInitializations(Initialization):
    """
    Combination of multiple initializers.
    """

    def __init__(self, initializations):
        """
        Constructor.

        :param initializations: list of initializations
        :type initializations: [Initializations]
        """

        assert isinstance(initializations, list)
        assert len(initializations) > 0
        for initialization in initializations:
            assert isinstance(initialization, Initialization)

        self.initializations = initializations
        """ ([Initializations]) Initializations. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        for initialization in self.initializations:
            initialization(images, perturbations)


class LInfUniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_inf ball.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: epsilon to project on
        :type epsilon: float
        """

        self.epsilon = epsilon
        """ (float) Epsilon. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=float('inf')).reshape(perturbations.size()).astype(numpy.float32))


class SmoothInitialization(Initialization):
    """
    Gaussian smoothing as initialization; can be used after any random initialization; does not enforce any cosntraints.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        sigma = numpy.random.uniform(1, 2)
        gamma = numpy.random.uniform(5, 30)
        gaussian_smoothing = common.torch.GaussianLayer(sigma=sigma, channels=perturbations.size()[1])
        if common.torch.is_cuda(perturbations):
            gaussian_smoothing = gaussian_smoothing.cuda()
        perturbations.data = 1 / (1 + torch.exp(-gamma * (gaussian_smoothing.forward(perturbations) - 0.5)))