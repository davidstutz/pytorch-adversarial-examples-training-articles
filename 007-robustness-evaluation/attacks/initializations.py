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


class ZeroInitialization(Initialization):
    """
    Zero initialization.
    """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        perturbations.data.zero_()


class L2UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_2 ball.
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

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=2).reshape(perturbations.size()).astype(numpy.float32))


class L2UniformSphereInitialization(Initialization):
    """
    Uniform initialization on L_2 sphere.
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

        perturbations.data = torch.from_numpy(common.numpy.uniform_sphere(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=2).reshape(perturbations.size()).astype(numpy.float32))


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


class LInfUniformSphereInitialization(Initialization):
    """
    Uniform initialization on L_inf sphere.
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

        perturbations.data = torch.from_numpy(common.numpy.uniform_sphere(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=float('inf')).reshape(perturbations.size()).astype(numpy.float32))


class L1UniformNormInitialization(Initialization):
    """
    Uniform initialization, wrt. norm and direction, in L_1 ball.
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

        perturbations.data = torch.from_numpy(common.numpy.uniform_norm(perturbations.size()[0], numpy.prod(perturbations.size()[1:]), epsilon=self.epsilon, ord=1).reshape(perturbations.size()).astype(numpy.float32))
