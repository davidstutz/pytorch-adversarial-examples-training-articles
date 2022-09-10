import torch
from .classifier import Classifier
from .resnet_block import *
from .utils import *


class ResNet(Classifier):
    """
    Simple classifier.
    """

    # resolution = C x H x W
    def __init__(self, N_class, resolution=(1, 32, 32), blocks=[3, 3, 3], block='', normalization='bn', channels=64, zero_residuals=True, **kwargs):
        """
        Initialize classifier.

        :param N_class: number of classes to classify
        :type N_class: int
        :param resolution: resolution (assumed to be square)
        :type resolution: int
        :param blocks: layers per block
        :type blocks: [int]
        :param normalization: normalization to use
        :type normalization: None or torch.nn.Module
        :param channels: channels to start with
        :type channels: int
        """

        super(ResNet, self).__init__(N_class, resolution, **kwargs)
        assert len(blocks) > 0
        assert block in ['', 'bottleneck']

        self.blocks = blocks
        """ ([int]) Blocks. """

        self.block = block
        """ (object) Block. """
        block_class = ResNetBlock
        if self.block == 'bottleneck':
            block_class = ResNetBottleneckBlock

        self.channels = channels
        """ (int) Channels. """

        self.normalization = normalization
        """ (str) Normalization. """

        self.zero_residuals = zero_residuals
        """ (booL) Zero residuals. """

        activation = 'relu'
        conv1 = torch.nn.Conv2d(self.resolution[0], self.channels, kernel_size=3, stride=1, padding=1, bias=self.include_bias)
        torch.nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity=activation)
        if self.include_bias:
            torch.nn.init.constant_(conv1.bias, 0)
        self.append_layer('conv1', conv1)

        norm1 = get_normalization2d(self.normalization, self.channels)
        self.append_layer('%s1' % self.normalization, norm1)

        relu = torch.nn.ReLU(inplace=True)
        self.append_layer('relu1', relu)

        downsampled = 1
        expanded_planes = self.channels
        for i in range(len(self.blocks)):
            #in_planes = (2 ** max(0, i - 1)) * self.channels
            out_planes = (2 ** i) * self.channels
            layers = self.blocks[i]
            stride = 2 if i > 0 else 1

            downsample = None
            if stride != 1 or expanded_planes != out_planes * block_class.expansion:
                conv = torch.nn.Conv2d(expanded_planes, out_planes * block_class.expansion, kernel_size=1, stride=stride, bias=self.include_bias)
                torch.nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity=activation)
                if self.include_bias:
                    torch.nn.init.constant_(conv.bias, 0)

                norm = get_normalization2d(normalization, out_planes * block_class.expansion)
                downsample = torch.nn.Sequential(*[conv, norm])

            sequence = []
            sequence.append(block_class(expanded_planes, out_planes, stride=stride, downsample=downsample, normalization=self.normalization,
                                        include_bias=self.include_bias, zero_residuals=self.zero_residuals))
            expanded_planes = out_planes * block_class.expansion
            for _ in range(1, layers):
                sequence.append(block_class(expanded_planes, out_planes, stride=1, downsample=None, normalization=self.normalization,
                                            include_bias=self.include_bias, zero_residuals=self.zero_residuals))

            self.append_layer('block%d' % i, torch.nn.Sequential(*sequence))
            downsampled *= stride

        representation = out_planes * block_class.expansion
        pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.append_layer('avgpool', pool)

        view = ViewOrReshape(-1, representation)
        self.append_layer('view', view)

        logits = torch.nn.Linear(representation, self._N_output, bias=self.include_bias)
        torch.nn.init.kaiming_normal_(logits.weight, nonlinearity=activation)
        if self.include_bias:
            torch.nn.init.constant_(logits.bias, 0)
        self.append_layer('logits', logits)

    def __str__(self):
        """
        Print network.
        """

        string = super(ResNet, self).__str__()
        string += '(blocks: %s)\n' % '-'.join(list(map(str, self.blocks)))
        string += '(block: %s)\n' % self.block
        string += '(channels: %d)\n' % self.channels
        string += '(normalization: %s)\n' % self.normalization
        string += '(zero_residuals: %s)\n' % self.zero_residuals

        return string