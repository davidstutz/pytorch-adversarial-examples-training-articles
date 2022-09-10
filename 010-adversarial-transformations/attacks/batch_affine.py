import torch
from .attack import Attack
import common.torch
import models
import numpy


class BatchAffine(Attack):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self, optimizer, **kwargs):
        """
        Constructor.

        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        """

        super(BatchAffine, self).__init__()

        self.optimizer = optimizer
        """ (torch.optim.Optimizer) Optimizer. """

        self.kwargs = kwargs
        """ (dict) Keyword arguments for optimizer. """

        self.thetas = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.N_theta = None
        """ (int) Number of transformations. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.c = None
        """ (float) Weight of norm. """

        self.normalized = False
        """ (bool) Normalize gradients. """

        self.norm = None
        """ (Norm) Norm. """

        self.projection = None
        """ (Projection) Projection. """

        self.initialization = None
        """ (Initialization) Initializer. """

        self.interpolation_mode = 'bilinear'
        """ (str) Decoder interpolation mode. """

        self.padding_mode = 'reflection'
        """ (str) Decoder padding mode. """

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

        super(BatchAffine, self).run(model, images, objective, writer, prefix)

        assert self.max_iterations is not None
        assert self.c is not None
        assert self.norm is not None
        is_cuda = common.torch.is_cuda(model)

        self.thetas = torch.from_numpy(numpy.zeros((images.size(0), self.N_theta), dtype=numpy.float32))
        if self.initialization is not None:
            self.initialization(None, self.thetas)
        if is_cuda:
            self.thetas = self.thetas.cuda()

        batch_size = self.thetas.size()[0]
        success_errors = numpy.ones((batch_size), dtype=numpy.float32)*1e12
        success_perturbations = numpy.zeros(images.size(), dtype=numpy.float32)

        self.thetas = torch.autograd.Variable(self.thetas, requires_grad=True)
        optimizer = self.optimizer([self.thetas], **self.kwargs)

        decoder = models.STNDecoder(self.N_theta, interpolation_mode=self.interpolation_mode, padding_mode=self.padding_mode)
        decoder.set_images(images)

        for i in range(self.max_iterations + 1):
            # MAIN LOOP OF ATTACK
            # ORDER IMPORTANT

            # Zero the gradient, as they are acculumated in PyTorch!
            optimizer.zero_grad()

            # 0/
            # Projections if necessary.
            if self.projection is not None:
                self.projection(torch.zeros_like(self.thetas), self.thetas)

            # 1/
            # Forward pass.
            perturbed_images = decoder(self.thetas)
            output_logits = model(perturbed_images)

            # 2/
            # Compute the error.
            error = self.c*self.norm(self.thetas) + objective(output_logits, self.thetas)

            # 3/
            # Check for improvement.
            norm = self.norm(self.thetas)
            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b].cpu().item()
                    success_perturbations[b] = (perturbed_images[b] - images[b]).detach().cpu()

            # 4/
            # Backward pass.
            loss = torch.sum(error)
            loss.backward()

            successes = objective.success(output_logits)
            true_confidences = objective.true_confidence(output_logits)
            target_confidences = objective.target_confidence(output_logits)

            for b in range(batch_size):
                writer.add_scalar('%ssuccess_%d' % (prefix, b), successes[b], global_step=i)
                writer.add_scalar('%strue_confidence_%d' % (prefix, b), true_confidences[b], global_step=i)
                writer.add_scalar('%starget_confidence_%d' % (prefix, b), target_confidences[b], global_step=i)
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)
                writer.add_scalar('%snorm_%d' % (prefix, b), norm[b], global_step=i)
            if self.progress is not None:
                self.progress('attack', i, self.max_iterations, info='success=%g error=%.2f norm=%g' % (
                    torch.sum(successes).item(),
                    torch.mean(error).item(),
                    torch.mean(norm).item(),
                ), width=10)


            # Quick hack for handling the last iteration correctly.
            if i == self.max_iterations:
                break

            # 6/
            # Get gradients, normalize and add momentum.
            if self.normalized:
                self.norm.normalize(self.thetas.grad)

            optimizer.step()

        return success_perturbations, success_errors

