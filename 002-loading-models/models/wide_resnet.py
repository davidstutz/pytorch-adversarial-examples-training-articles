import torch
from .classifier import Classifier
from .wide_resnet_block import *
from .utils import *


class WideResNet(Classifier):
    """
    Wide Res-Net taken from https://github.com/meliketoy/wide-resnet.pytorch
    """

    def __init__(self, N_class, resolution=(1, 32, 32), depth=28, width=10, normalization='bn', channels=16, dropout=False, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param depth: depth from which to calculate the blocks
        :type depth: int
        :param depth: width factor
        :type depth: int
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        :param dropout: dropout rate
        :type dropout: float
        """

        super(WideResNet, self).__init__(N_class, resolution, **kwargs)

        self.depth = depth
        """ (int) Depth. """

        self.width = width
        """ (int) Width. """

        self.channels = channels
        """ (int) Channels. """

        self.dropout = dropout
        """ (int) Dropout. """

        self.normalization = normalization
        """ (callable) Normalization. """

        self.in_planes = channels
        """ (int) Helper for channels. """

        assert (depth-4)%6 == 0, 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = width

        planes = [self.channels, self.channels*k, 2*self.channels*k, 4*self.channels*k]

        activation = 'relu'
        downsampled = 1
        conv = torch.nn.Conv2d(resolution[0], planes[0], kernel_size=3, stride=1, padding=1, bias=self.include_bias)
        #torch.nn.init.xavier_uniform_(conv.weight, gain=numpy.sqrt(2))
        torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity=activation)
        if self.include_bias:
            torch.nn.init.constant_(conv.bias, 0)
        self.append_layer('conv0', conv)

        block1 = self._wide_layer(WideResNetBlock, planes[1], n, stride=1)
        self.append_layer('block1', block1)
        block2 = self._wide_layer(WideResNetBlock, planes[2], n, stride=2)
        downsampled *= 2
        self.append_layer('block2', block2)
        block3 = self._wide_layer(WideResNetBlock, planes[3], n, stride=2)
        downsampled *= 2
        self.append_layer('block3', block3)

        norm3 = get_normalization2d(self.normalization, self.in_planes)
        self.append_layer('%s3' % self.normalization, norm3)

        relu = torch.nn.ReLU(inplace=True)
        self.append_layer('relu3', relu)

        representation = planes[3]
        pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.append_layer('avgpool', pool)

        view = ViewOrReshape(-1, representation)
        self.append_layer('view', view)

        logits = torch.nn.Linear(planes[3], self._N_output, bias=self.include_bias)
        torch.nn.init.kaiming_normal_(logits.weight, nonlinearity=activation)
        if self.include_bias:
            torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

    def _wide_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1]*(blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.dropout, self.normalization, include_bias=self.include_bias))
            self.in_planes = planes

        return torch.nn.Sequential(*layers)

    def __str__(self):
        """
        Print network.
        """

        string = super(WideResNet, self).__str__()
        string += '(depth: %d)\n' % self.depth
        string += '(width: %d)\n' % self.width
        string += '(channels: %d)\n' % self.channels
        string += '(normalization: %s)\n' % self.normalization
        string += '(dropout: %s)\n' % self.dropout

        return string