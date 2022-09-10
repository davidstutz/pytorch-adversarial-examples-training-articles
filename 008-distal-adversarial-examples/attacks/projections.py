import common.torch
import torch


class Projection:
    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        raise NotImplementedError()


class SequentialProjections(Projection):
    def __init__(self, projections):
        """
        Constructor.

        :param projections: list of projections
        :type projections: [Projection]
        """

        assert isinstance(projections, list)
        assert len(projections) > 0
        for projection in projections:
            assert isinstance(projection, Projection)

        self.projections = projections
        """ ([Projection]) Projections. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        for projection in self.projections:
            projection(images, perturbations)


class BoxProjection(Projection):
    def __init__(self, min_bound=0, max_bound=1):
        """
        Constructor.

        :param min_bound: minimum bound
        :param min_bound: float
        :param max_bound: maximum bound
        :type: max_bound: float
        """

        assert isinstance(min_bound, float) or isinstance(min_bound, int) or isinstance(min_bound, torch.Tensor) or min_bound is None
        assert isinstance(max_bound, float) or isinstance(max_bound, int) or isinstance(max_bound, torch.Tensor) or max_bound is None

        self.min_bound = min_bound
        """ (float) Minimum bound. """

        self.max_bound = max_bound
        """ (float) Maximum bound. """

    def __call__(self, images, perturbations):
        """
        Projection.

        :param images: images
        :type images: torch.autograd.Variable
        :param perturbations: perturbations
        :type perturbations: torch.autograd.Variable
        """

        if self.max_bound is not None:
            if isinstance(self.max_bound, torch.Tensor):
                # self.max_bound should be 1 x C x H x W for image or in general 1 x anything
                # so it can be repeated along the first dimension
                assert self.max_bound.size(0) == 1
                assert common.torch.is_cuda(self.max_bound) == common.torch.is_cuda(perturbations)

                # workaround to have proper repeating
                repeat = [perturbations.size(0)]
                for i in range(1, len(list(self.max_bound.size()))):
                    repeat.append(1)
                max_bound = self.max_bound.repeat(repeat).float()

                perturbations.data = torch.min(max_bound, perturbations.data)
            else:
                perturbations.data = torch.min(torch.ones_like(perturbations.data) * self.max_bound - images.data, perturbations.data)
        if self.min_bound is not None:
            if isinstance(self.min_bound, torch.Tensor):
                # self.max_bound should be 1 x C x H x W for image or in general 1 x anything
                # so it can be repeated along the first dimension
                assert self.min_bound.size(0) == 1
                assert common.torch.is_cuda(self.min_bound) == common.torch.is_cuda(perturbations)

                # workaround to have proper repeating
                repeat = [perturbations.size(0)]
                for i in range(1, len(list(self.min_bound.size()))):
                    repeat.append(1)
                min_bound = self.min_bound.repeat(repeat).float()

                perturbations.data = torch.max(min_bound, perturbations.data)
            else:
                perturbations.data = torch.max(torch.ones_like(perturbations.data)*self.min_bound - images.data, perturbations.data)


class LInfProjection(Projection):
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

        perturbations.data = common.torch.project_ball(perturbations.data, self.epsilon, ord=float('inf'))

