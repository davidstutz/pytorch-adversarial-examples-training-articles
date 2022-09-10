import torch
import numpy
from .attack import *
from common.log import log
import common.torch


class BatchGradientDescent(Attack):
    """
    Implementation of untargetetd PGD attack.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(BatchGradientDescent, self).__init__()

        self.perturbations = None
        """ (torch.autograd.Variable) Perturbation of attack. """

        self.max_iterations = None
        """ (int) Maximum number of iterations. """

        self.c = None
        """ (float) Weight of norm. """

        self.base_lr = None
        """ (float) Base learning rate. """

        self.lr_factor = None
        """ (float) Learning rate decay. """

        self.momentum = None
        """ (float) Momentum. """

        self.backtrack = False
        """ (bool) Backtrack. """

        self.normalized = False
        """ (bool) Normalize gradients. """

        self.scaled = False
        """ (bool) Normalize gradients. """

        self.norm = None
        """ (Norm) Norm. """

        self.initialization = None
        """ (Initialization) Initializer. """

        self.projection = None
        """ (Projection) Projection. """

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

        super(BatchGradientDescent, self).run(model, images, objective, writer, prefix)

        assert not (self.normalized and self.scaled)
        assert self.max_iterations is not None
        assert self.c is not None
        assert self.base_lr is not None
        assert self.lr_factor is not None
        assert self.momentum is not None
        assert self.norm is not None
        is_cuda = common.torch.is_cuda(model)

        self.perturbations = torch.from_numpy(numpy.zeros(images.size(), dtype=numpy.float32))
        if self.initialization is not None:
            self.initialization(images, self.perturbations)
        if is_cuda:
            self.perturbations = self.perturbations.cuda()

        batch_size = self.perturbations.size()[0]
        success_errors = numpy.ones((batch_size), dtype=numpy.float32)*1e12
        success_perturbations = numpy.zeros(self.perturbations.size(), dtype=numpy.float32)

        self.lrs = torch.from_numpy(numpy.ones(batch_size, dtype=numpy.float32) * self.base_lr)
        """ (numpy.ndarray) Holds per element learning rates. """

        if is_cuda:
            self.lrs = self.lrs.cuda()

        self.perturbations = torch.autograd.Variable(self.perturbations, requires_grad=True)
        self.gradients = torch.zeros_like(self.perturbations)
        """ (torch.autograd.Variable) Gradients. """

        for i in range(self.max_iterations + 1):
            # MAIN LOOP OF ATTACK
            # ORDER IMPORTANT
            
            # Zero the gradient, as they are acculumated in PyTorch!
            if i > 0:
                self.perturbations.grad.data.zero_()

            # 0/
            # Projections if necessary.
            if self.projection is not None:
                self.projection(images, self.perturbations)

            # 1/
            # Forward pass.
            output_logits = model(images + self.perturbations)
            assert not torch.any(torch.isnan(self.perturbations))
            assert not torch.any(torch.isnan(output_logits))

            # 2/
            # Compute the error.
            error = self.c*self.norm(self.perturbations) + objective(output_logits, self.perturbations)

            # 3/
            # Check for improvement.
            norm = self.norm(self.perturbations)
            for b in range(batch_size):
                if error[b].item() < success_errors[b]:
                    success_errors[b] = error[b].cpu().item()
                    success_perturbations[b] = self.perturbations[b].detach().cpu().numpy()

            # 4/
            # Backward pass.
            torch.sum(error).backward()

            # 5/
            # Gradient satistics and logging.
            gradients = self.perturbations.grad.clone()
            #assert not torch.any(torch.isnan(gradients))
            gradient_magnitudes = torch.mean(torch.abs(gradients.view(batch_size, -1)), dim=1)/self.perturbations.size()[0]

            successes = objective.success(output_logits)
            true_confidences = objective.true_confidence(output_logits)
            target_confidences = objective.target_confidence(output_logits)

            for b in range(batch_size):
                writer.add_scalar('%ssuccess_%d' % (prefix, b), successes[b], global_step=i)
                writer.add_scalar('%strue_confidence_%d' % (prefix, b), true_confidences[b], global_step=i)
                writer.add_scalar('%starget_confidence_%d' % (prefix, b), target_confidences[b], global_step=i)
                writer.add_scalar('%slr_%d' % (prefix, b), self.lrs[b], global_step=i)
                writer.add_scalar('%serror_%d' % (prefix, b), error[b], global_step=i)
                writer.add_scalar('%snorm_%d' % (prefix, b), norm[b], global_step=i)
                writer.add_scalar('%sgradient_%d' % (prefix, b), gradient_magnitudes[b], global_step=i)
            if self.progress is not None:
                self.progress('attack', i, self.max_iterations, info='success=%g error=%.2f norm=%g lr=%g' % (
                    torch.sum(successes).item(),
                    torch.mean(error).item(),
                    torch.mean(norm).item(),
                    torch.mean(self.lrs).item(),
                ), width=10)

            # Quick hack for handling the last iteration correctly.
            if i == self.max_iterations:
                break

            # 6/
            # Get gradients, normalize and add momentum.
            if self.normalized:
                self.norm.normalize(gradients)
                #assert not torch.any(torch.isnan(gradients))
            elif self.scaled:
                self.norm.scale(gradients)
                #assert not torch.any(torch.isnan(gradients))

            self.gradients.data = self.momentum*self.gradients.data + (1 - self.momentum)*gradients.data
            #assert not torch.any(torch.isnan(self.gradients))

            # 7/ Update.
            if self.backtrack:
                next_perturbations = self.perturbations - torch.mul(common.torch.expand_as(self.lrs, self.gradients), self.gradients)
                #assert not torch.any(torch.isnan(self.lrs))
                #assert not torch.any(torch.isnan(next_perturbations))

                if self.projection is not None:
                    self.projection(images, next_perturbations)

                next_output_logits = model(images + next_perturbations)
                next_error = self.c * self.norm(next_perturbations) + objective(next_output_logits, next_perturbations)

                # Update learning rate if requested.
                for b in range(batch_size):
                    if next_error[b].item() <= error[b]:
                        self.perturbations[b].data -= self.lrs[b]*self.gradients[b].data
                    else:
                        self.lrs[b] = max(self.lrs[b] / self.lr_factor, 1e-20)
            else:
                self.perturbations.data -= torch.mul(common.torch.expand_as(self.lrs, self.gradients), self.gradients)

        return success_perturbations, success_errors

