import torch
import numpy
import common.torch
import common.summary
import common.numpy
from common.progress import ProgressBar


class NormalTraining:
    """
    Normal training.
    """

    def __init__(self, model, trainset, testset, optimizer, scheduler, augmentation=None, loss=common.torch.classification_loss, writer=common.summary.SummaryWriter(), cuda=False):
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
        :param augmentation: augmentation
        :type augmentation: imgaug.augmenters.Sequential
        :param writer: summary writer
        :type writer: torch.utils.tensorboard.SummaryWriter or TensorboardX equivalent
        :param cuda: run on CUDA device
        :type cuda: bool
        """

        assert loss is not None
        assert callable(loss)
        assert isinstance(model, torch.nn.Module)
        assert len(trainset) > 0
        assert len(testset) > 0
        assert isinstance(trainset, torch.utils.data.DataLoader)
        assert isinstance(testset, torch.utils.data.DataLoader)
        assert isinstance(trainset.sampler, torch.utils.data.RandomSampler)
        assert isinstance(testset.sampler, torch.utils.data.SequentialSampler)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler)
        assert (cuda and common.torch.is_cuda(model)) or (not cuda and not common.torch.is_cuda(model))

        self.writer = writer
        """ (torch.util.tensorboard.SummaryWriter or equivalent) Summary writer. """

        self.progress = ProgressBar()
        """ (Timer) """

        self.model = model
        """ (torch.nn.Module) Model. """

        self.layers = range(len(list(model.parameters())))
        """ ([int]) Layers for projection. """

        self.trainset = trainset
        """ (torch.utils.data.DatLoader) Taining set. """

        self.testset = testset
        """ (torch.utils.data.DatLoader) Test set. """

        self.optimizer = optimizer
        """ (torch.optim.Optimizer) Optimizer. """

        self.scheduler = scheduler
        """ (torch.optim.LRScheduler) Scheduler. """

        self.augmentation = augmentation
        """ (imgaug.augmenters.Sequential) Augmentation. """

        self.cuda = cuda
        """ (bool) Run on CUDA. """

        self.loss = loss
        """ (callable) Classificaiton loss. """

        self.summary_histograms = False
        """ (bool) Summary for histograms. """

        self.writer.add_text('config/model', self.model.__class__.__name__)
        self.writer.add_text('config/model_details', str(self.model))
        self.writer.add_text('config/trainset', self.trainset.dataset.__class__.__name__)
        self.writer.add_text('config/testset', self.testset.dataset.__class__.__name__)
        self.writer.add_text('config/optimizer', self.optimizer.__class__.__name__)
        self.writer.add_text('config/scheduler', self.scheduler.__class__.__name__)
        self.writer.add_text('config/cuda', str(self.cuda))

        self.writer.add_text('model', str(self.model))
        self.writer.add_text('optimizer', str(common.summary.to_dict(self.optimizer)))
        self.writer.add_text('scheduler', str(common.summary.to_dict(self.scheduler)))

    def quantize(self):
        """
        Quantization.
        """

        if self.quantization is not None:
            forward_model, contexts = common.quantization.quantize(self.quantization, self.model)
            return forward_model, contexts
        return self.model, None

    def project(self):
        """
        Projection.
        """

        if self.projection is not None:
            self.projection(None, self.model, self.layers)

    def train(self, epoch):
        """
        Training step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.train()
        assert self.model.training is True

        for b, (inputs, targets) in enumerate(self.trainset):
            if self.augmentation is not None:
                inputs = self.augmentation.augment_images(inputs.numpy())

            inputs = common.torch.as_variable(inputs, self.cuda)
            #inputs = inputs.permute(0, 3, 1, 2)
            assert len(targets.shape) == 1
            targets = common.torch.as_variable(targets, self.cuda)
            assert len(list(targets.size())) == 1

            self.optimizer.zero_grad()
            logits = self.model(inputs)
            loss = self.loss(logits, targets)
            error = common.torch.classification_error(logits, targets)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            global_step = epoch * len(self.trainset) + b
            self.writer.add_scalar('train/lr', self.scheduler.get_lr()[0], global_step=global_step)
            self.writer.add_scalar('train/loss', loss.item(), global_step=global_step)
            self.writer.add_scalar('train/error', error.item(), global_step=global_step)
            self.writer.add_scalar('train/confidence', torch.mean(torch.max(common.torch.softmax(logits, dim=1), dim=1)[0]).item(), global_step=global_step)
            #print(loss.item(), error.item())

            if self.summary_histograms:
                self.writer.add_histogram('train/logits', torch.max(logits, dim=1)[0], global_step=global_step)
                self.writer.add_histogram('train/confidences', torch.max(common.torch.softmax(logits, dim=1), dim=1)[0], global_step=global_step)

            j = 0
            for parameter in self.model.parameters():
                self.writer.add_scalar('train/weight/%d' % j, torch.mean(torch.abs(parameter.data)).item(), global_step=global_step)
                self.writer.add_scalar('train/gradient/%d' % j, torch.mean(torch.abs(parameter.grad.data)).item(), global_step=global_step)
                if self.summary_histograms:
                    self.writer.add_histogram('train/weights/%d' % j, parameter.view(-1), global_step=global_step)
                    self.writer.add_histogram('train/gradients/%d' % j, parameter.grad.view(-1), global_step=global_step)
                j += 1

            self.writer.add_images('train/images', inputs[:16], global_step=global_step)
            self.progress('train %d' % epoch, b, len(self.trainset), info='loss=%g error=%g lr=%g' % (
                loss.item(),
                error.item(),
                self.scheduler.get_lr()[0],
            ))

    def test(self, epoch):
        """
        Test step.

        :param epoch: epoch
        :type epoch: int
        """

        self.model.eval()
        assert self.model.training is False

        # reason to repeat this here: use correct loss for statistics
        losses = None
        errors = None
        logits = None
        probabilities = None

        for b, (inputs, targets) in enumerate(self.testset):
            inputs = common.torch.as_variable(inputs, self.cuda)
            #inputs = inputs.permute(0, 3, 1, 2)
            targets = common.torch.as_variable(targets, self.cuda)

            outputs = self.model(inputs)
            b_losses = self.loss(outputs, targets, reduction='none')
            b_errors = common.torch.classification_error(outputs, targets, reduction='none')

            losses = common.numpy.concatenate(losses, b_losses.detach().cpu().numpy())
            errors = common.numpy.concatenate(errors, b_errors.detach().cpu().numpy())
            logits = common.numpy.concatenate(logits, torch.max(outputs, dim=1)[0].detach().cpu().numpy())
            probabilities = common.numpy.concatenate(probabilities, common.torch.softmax(outputs, dim=1).detach().cpu().numpy())

            self.progress('test %d' % epoch, b, len(self.testset), info='loss=%g error=%g' % (
                torch.mean(b_losses).item(),
                torch.mean(b_errors.float()).item()
            ))

        confidences = numpy.max(probabilities, axis=1)
        global_step = epoch  # epoch * len(self.trainset) + len(self.trainset) - 1

        self.writer.add_scalar('test/loss', numpy.mean(losses), global_step=global_step)
        self.writer.add_scalar('test/error', numpy.mean(errors), global_step=global_step)
        self.writer.add_scalar('test/logit', numpy.mean(logits), global_step=global_step)
        self.writer.add_scalar('test/confidence', numpy.mean(confidences), global_step=global_step)
        #log('test %d: error=%g loss=%g' % (epoch, numpy.mean(errors), numpy.mean(losses)))

        if self.summary_histograms:
            self.writer.add_histogram('test/losses', losses, global_step=global_step)
            self.writer.add_histogram('test/errors', errors, global_step=global_step)
            self.writer.add_histogram('test/logits', logits, global_step=global_step)
            self.writer.add_histogram('test/confidences', confidences, global_step=global_step)

        return probabilities

    def step(self, epoch):
        """
        Training + test step.

        :param epoch: epoch
        :type epoch: int
        :return: probabilities of test set
        :rtype: numpy.array
        """

        self.train(epoch)
        return self.test(epoch)