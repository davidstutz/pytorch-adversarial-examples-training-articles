import torch
from .resnet_block import ResNetBlock
from .wide_resnet_block import WideResNetBlock
from .utils import *


class Classifier(torch.nn.Module):
    """
    Simple classifier.
    """

    def __init__(self, N_class, resolution, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution
        :type resolution: [int]
        """

        super(Classifier, self).__init__()

        assert N_class > 0, 'positive N_class expected'
        assert len(resolution) <= 3

        self.N_class = int(N_class)  # Having strange bug where torch complaints about numpy.in64 being passed to nn.Linear.
        """ (int) Number of classes. """

        self.resolution = list(resolution)
        """ ([int]) Resolution as (channels, height, width) """

        # __ attributes are private, which is important for the State to work properly.
        self.__layers = []
        """ ([str]) Will hold layer names. """

        self.kwargs = kwargs
        """ (dict) Kwargs. """

        self.include_clamp = self.kwargs.get('clamp', True)
        """ (bool) Clamp. """

        self.include_whiten = self.kwargs.get('whiten', False)
        """ (bool) Whiten. """

        self.include_bias = self.kwargs.get('bias', True)
        """ (bool) Bias. """

        self._N_output = self.N_class if self.N_class > 2 else 1
        """ (int) Number of outputs. """

        if self.include_clamp:
            self.append_layer('clamp',  Clamp())

        assert not (self.include_whiten and self.include_rescale)

        if self.include_whiten:
            whiten = Normalize(resolution[0])
            self.append_layer('whiten', whiten)
            whiten.weight.requires_grad = False
            whiten.bias.requires_grad = False

    def append_layer(self, name, layer):
        """
        Add a layer.

        :param name: layer name
        :type name: str
        :param layer: layer
        :type layer: torch.nn.Module
        """

        setattr(self, name, layer)
        self.__layers.append(name)

    def forward(self, image):
        """
        Forward pass, takes an image and outputs the predictions.

        :param image: input image
        :type image: torch.autograd.Variable
        :return: logits
        :rtype: torch.autograd.Variable
        """

        output = image

        if self.include_whiten:
            assert self.whiten.weight.requires_grad is False
            assert self.whiten.bias.requires_grad is False

        for name in self.__layers:
            output = getattr(self, name)(output)
        return output

    def __str__(self):
        """
        Print network.
        """

        string = '(N_class: %d)\n' % self.N_class
        string += '(resolution: %s)\n' % 'x'.join(list(map(str, self.resolution)))
        string += '(include_bias: %s)\n' % self.include_bias
        string += '(include_clamp: %s)\n' % self.include_clamp
        string += '(include_whiten: %s)\n' % self.include_whiten
        if self.include_whiten:
            string += '\t(weight=%s)\n' % self.whiten.weight.data.detach().cpu().numpy()
            string += '\t(bias=%s)\n' % self.whiten.bias.data.detach().cpu().numpy()

        def module_description(module):
            string = '(' + name + ', ' + module.__class__.__name__
            weight = getattr(module, 'weight', None)

            if weight is not None:
                string += ', weight=%g,%g+-%g,%g' % (torch.min(weight).item(), torch.mean(weight).item(), torch.std(weight).item(), torch.max(weight).item())
            bias = getattr(module, 'bias', None)
            if bias is not None:
                string += ', bias=%g,%g+-%g,%g' % (torch.min(bias).item(), torch.mean(bias).item(), torch.std(bias).item(), torch.max(bias).item())
            string += ')\n'

            return string

        for name in self.__layers:
            module = getattr(self, name)
            string += module_description(module)

            if isinstance(getattr(self, name), torch.nn.Sequential) or isinstance(getattr(self, name), ResNetBlock) or isinstance(getattr(self, name), WideResNetBlock):
                for module in getattr(self, name).modules():
                    string += '\t' + module_description(module)
        return string

