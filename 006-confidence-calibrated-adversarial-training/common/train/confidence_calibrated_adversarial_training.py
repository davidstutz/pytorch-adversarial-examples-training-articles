import torch
import common.torch
import common.summary
import common.numpy
from .adversarial_training import *


class ConfidenceCalibratedAdversarialTraining(AdversarialTraining):
    """
    Confidence-calibrated adversarial training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, attack, objective, loss, transition, fraction=0.5, augmentation=None, writer=common.summary.SummaryWriter(), cuda=False):
        """
        Constructor.

        :param model: model
        :type model: torch.nn.Module
        :param trainset: training set
        :type trainset: torch.utils.data.DataLoader
        :param testset: test set
        :type testset: torch.utils.data.DataLoader
        :param optimizer: optimizer
        :type optimizer: torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: torch.optim.LRScheduler
        :param attack: attack
        :type attack: attacks.Attack
        :param objective: objective
        :type objective: attacks.Objective
        :param loss: loss
        :type loss: callable
        :param loss: transition
        :type loss: callable
        :param fraction: fraction of adversarial examples per batch
        :type fraction: float
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        super(ConfidenceCalibratedAdversarialTraining, self).__init__(model, trainset, testset, optimizer, scheduler, attack, objective, fraction, augmentation, loss, writer, cuda)

        self.loss = loss
        """ (callable) Loss. """

        self.transition = transition
        """ (callable) Transition. """

        self.writer.add_text('config/loss', self.loss.__name__)
        self.writer.add_text('config/transition', self.transition.__name__)

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """
        assert self.fraction < 1

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                inputs = self.augmentation.augment_images(inputs.numpy())

            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            distributions = common.torch.one_hot(targets, self.model.N_class)

            split = int(self.fraction * inputs.size()[0])
            # update fraction for correct loss computation
            fraction = split / float(inputs.size(0))

            clean_inputs = inputs[:split]
            adversarial_inputs = inputs[split:]
            clean_targets = targets[:split]
            adversarial_targets = targets[split:]
            clean_distributions = distributions[:split]
            adversarial_distributions = distributions[split:]

            self.model.eval()
            self.objective.set(adversarial_targets)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, adversarial_inputs, self.objective)
            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            adversarial_inputs = adversarial_inputs + adversarial_perturbations

            gamma, adversarial_norms = self.transition(adversarial_perturbations)
            gamma = common.torch.expand_as(gamma, adversarial_distributions)

            adversarial_distributions = adversarial_distributions*(1 - gamma)
            adversarial_distributions += gamma*torch.ones_like(adversarial_distributions)/self.model.N_class

            inputs = torch.cat((clean_inputs, adversarial_inputs), dim=0)

            self.optimizer.zero_grad()
            self.model.train()
            logits = self.model(inputs)
            clean_logits = logits[:split]
            adversarial_logits = logits[split:]

            adversarial_loss = self.loss(adversarial_logits, adversarial_distributions)
            adversarial_error = common.torch.classification_error(adversarial_logits, adversarial_targets)

            clean_loss = self.loss(clean_logits, clean_distributions)
            clean_error = common.torch.classification_error(clean_logits, clean_targets)
            loss = (1 - fraction) * clean_loss + fraction * adversarial_loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)

            self.writer.add_scalar('train/loss', clean_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', clean_error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0]).item(), global_step=global_step)

            self.writer.add_histogram('train/logits', torch.max(clean_logits, dim=1)[0], global_step=global_step)
            self.writer.add_histogram('train/confidences', torch.max(torch.nn.functional.softmax(clean_logits, dim=1), dim=1)[0], global_step=global_step)

            success = torch.clamp(torch.abs(adversarial_targets - torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[1]), max=1)
            self.writer.add_scalar('train/adversarial_loss', adversarial_loss.item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_error', adversarial_error.item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_confidence', torch.mean(torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[0]).item(), global_step=global_step)
            self.writer.add_scalar('train/adversarial_success', torch.mean(success.float()).item(), global_step=global_step)

            self.writer.add_histogram('train/adversarial_logits', torch.max(adversarial_logits, dim=1)[0], global_step=global_step)
            self.writer.add_histogram('train/adversarial_confidences', torch.max(torch.nn.functional.softmax(adversarial_logits, dim=1), dim=1)[0], global_step=global_step)

            self.writer.add_histogram('train/adversarial_objectives', adversarial_objectives, global_step=global_step)
            self.writer.add_histogram('train/adversarial_norms', adversarial_norms, global_step=global_step)

            self.writer.add_images('train/images', inputs[:min(16, split)], global_step=global_step)
            self.writer.add_images('train/adversarial_images', inputs[split:split + 16], global_step=global_step)

            self.progress(epoch, b, len(self.trainset), info='loss=%g error=%g advloss=%g adverr=%g' % (
                clean_loss,
                clean_error,
                adversarial_loss,
                adversarial_error,
            ))

    def test(self, epoch):
        """
        Test on adversarial examples.

        :param epoch: epoch
        :type epoch: int
        """

        self.optimizer.zero_grad()
        self.model.eval()
        assert self.model.training is False

        # reason to repeat this here: use correct loss for statistics
        losses = None
        errors = None
        logits = None
        probabilities = None

        for b, (inputs, targets) in enumerate(self.testset):
            if b >= 10:
                break;

            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            distributions = common.torch.as_variable(common.torch.one_hot(targets, self.model.N_class))

            with torch.no_grad():
                outputs = self.model(inputs)
                b_losses = self.loss(outputs, distributions, reduction='none').detach().cpu().numpy()
                b_errors = common.torch.classification_error(outputs, targets, reduction='none').float().detach().cpu().numpy()

                losses = common.numpy.concatenate(losses, b_losses)
                errors = common.numpy.concatenate(errors, b_errors)
                logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
                probabilities = common.numpy.concatenate(probabilities, common.torch.softmax(outputs, dim=1).detach().cpu().numpy())

                self.progress('test (test) %d' % epoch, b, len(self.testset), info='loss=%g error=%g' % (
                    numpy.mean(b_losses),
                    numpy.mean(b_errors)
                ))

        confidences = numpy.max(probabilities, axis=1)
        global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1

        self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)

        self.model.eval()
        losses = None
        errors = None
        logits = None
        confidences = None
        successes = None
        norms = None
        objectives = None

        for b, (inputs, targets) in enumerate(self.testset):
            if b >= self.max_batches:
                break

            inputs = common.torch.as_variable(inputs, self.cuda)
            targets = common.torch.as_variable(targets, self.cuda)
            distributions = common.torch.as_variable(common.torch.one_hot(targets, self.model.N_class))

            self.objective.set(targets)
            adversarial_perturbations, adversarial_objectives = self.attack.run(self.model, inputs, self.objective)
            objectives = common.numpy.concatenate(objectives, adversarial_objectives)

            adversarial_perturbations = common.torch.as_variable(adversarial_perturbations, self.cuda)
            inputs = inputs + adversarial_perturbations

            gamma, adversarial_norms = self.transition(adversarial_perturbations)
            gamma = common.torch.expand_as(gamma, distributions)
            distributions = distributions * (1 - gamma) + gamma * torch.ones_like(distributions) / self.model.N_class

            outputs = self.model(inputs)

            losses_b = self.loss(outputs, distributions, reduction='none').detach().cpu().numpy()
            errors_b = common.torch.classification_error(outputs, targets, reduction='none').detach().cpu().numpy()

            losses = common.numpy.concatenate(losses, losses_b)
            errors = common.numpy.concatenate(errors, errors_b)
            logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            confidences = common.numpy.concatenate(confidences, torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[0].detach().cpu().numpy())
            successes = common.numpy.concatenate(successes, torch.clamp(torch.abs(targets - torch.max(torch.nn.functional.softmax(outputs, dim=1), dim=1)[1]), max=1).detach().cpu().numpy())
            norms = common.numpy.concatenate(norms, adversarial_norms.detach().cpu().numpy())
            self.progress(epoch, b, self.max_batches, info='loss=%g error=%g' % (numpy.mean(losses_b), numpy.mean(errors_b)))

        global_step = epoch + 1# * len(self.trainset) + len(self.trainset) - 1
        self.writer.add_scalar('test/adversarial_loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/adversarial_error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/adversarial_logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/adversarial_confidence', numpy.mean(confidences), global_step=global_step)
        self.writer.add_scalar('test/adversarial_norm', numpy.mean(norms), global_step=global_step)
        self.writer.add_scalar('test/adversarial_objective', numpy.mean(objectives), global_step=global_step)
        self.writer.add_scalar('test/adversarial_success', numpy.mean(successes), global_step=global_step)

        self.writer.add_histogram('test/adversarial_losses', losses, global_step=global_step)
        self.writer.add_histogram('test/adversarial_errors', errors, global_step=global_step)
        self.writer.add_histogram('test/adversarial_logits', logits, global_step=global_step)
        self.writer.add_histogram('test/adversarial_confidences', confidences, global_step=global_step)
        self.writer.add_histogram('test/adversarial_norms', norms, global_step=global_step)
        self.writer.add_histogram('test/adversarial_objectives', objectives, global_step=global_step)

        return probabilities