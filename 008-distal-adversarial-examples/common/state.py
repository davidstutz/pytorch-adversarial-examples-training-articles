import torch
import os
from . import utils
from packaging import version


class State:
    """
    State of a model, including optional epoch and optimizer.
    """

    def __init__(self, model, optimizer=None, scheduler=None, epoch=None):
        """
        Constructor.

        :param model: model
        :type model: dict or torch.nn.Module
        :param optimizer: optimizer
        :type optimizer: dict or torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: dict or torch.optim.LRScheduler
        :param epoch: epoch
        :type epoch: int
        """

        self.model = model
        """ (dict or torch.nn.Module) Model. """

        self.optimizer = optimizer
        """ (dict or torch.optim.Optimizer) Optimizer. """

        self.scheduler = scheduler
        """ (dict or torch.optim.LRScheduler) Scheduler. """

        self.epoch = epoch
        """ (int) Epoch. """

    def save(self, filepath):
        """
        Save the state.

        :param filepath: file to save to
        :type filepath: str
        """

        model = self.model
        if not isinstance(model, dict):
            model = model.state_dict()
        model_class = self.model.__class__.__name__

        optimizer = self.optimizer
        if not isinstance(optimizer, dict) and optimizer is not None:
            optimizer = optimizer.state_dict()

        scheduler = self.scheduler
        if not isinstance(scheduler, dict) and scheduler is not None:
            scheduler = scheduler.state_dict()

        epoch = self.epoch
        assert utils.get_class('models', model_class) is not False
        arguments = dict((key, getattr(self.model, key)) for key in dir(self.model)
                         if not callable(getattr(self.model, key)) and not key.startswith('_') and not key == 'kwargs' and not key == 'T_destination')
        kwargs = getattr(self.model, 'kwargs', None)
        utils.makedir(os.path.dirname(filepath))

        data = {'model': model, 'model_class': model_class,
                'optimizer': optimizer, 'scheduler': scheduler,
                'epoch': epoch, 'arguments': arguments, 'kwargs': kwargs}
        if version.parse('1.6.0') < version.parse(torch.__version__):
            torch.save(data, filepath, pickle_protocol=2, _use_new_zipfile_serialization=False)
        else:
            torch.save(data, filepath)

    @staticmethod
    def checkpoint(filepath, model, optimizer=None, scheduler=None, epoch=None):
        """
        Quick access to State.save.

        :param filepath: path to file
        :type filepath: str
        :param model: model
        :type model: dict or torch.nn.Module
        :param optimizer: optimizer
        :type optimizer: dict or torch.optim.Optimizer
        :param scheduler: scheduler
        :type scheduler: dict or torch.optim.LRScheduler
        :param epoch: epoch
        :type epoch: int
        """

        state = State(model, optimizer, scheduler, epoch)
        state.save(filepath)

    @staticmethod
    def load(filepath):
        """
        Load a state.

        :param filepath: file to load
        :type filepath: str
        :return: state
        :rtype: State
        """

        assert os.path.exists(filepath), 'file %s not found' % str(filepath)

        # https://discuss.pytorch.org/t/gpu-memory-usage-increases-by-90-after-torch-load/9213/3
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)

        model_class = utils.get_class('models', checkpoint['model_class'])
        if 'kwargs' in checkpoint:
            arguments = {**checkpoint['arguments'], **checkpoint['kwargs']}
        else:
            arguments = {**checkpoint['arguments']}
        model = model_class(**arguments)
        model.load_state_dict(checkpoint['model'])

        state = State(model, checkpoint['optimizer'], checkpoint['scheduler'], checkpoint['epoch'])

        del checkpoint
        torch.cuda.empty_cache()

        return state