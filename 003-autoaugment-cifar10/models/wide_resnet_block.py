import torch
from .utils import *


class WideResNetBlock(torch.nn.Module):
    """
    Wide ResNet block taken from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py.
    """

    def __init__(self, inplanes, planes, stride=1, dropout=0, normalization='bn', include_bias=False):
        """
        Constructor.

        :param inplanes: input channels
        :type inplanes: int
        :param planes: output channels
        :type planes: int
        :param stride: stride
        :type stride: int
        :param dropout: dropout rate
        :type dropout: float
        :param normalization: whether to use normalization
        :type normalization: bool
        """

        assert inplanes > 0
        assert planes > 0
        assert planes >= inplanes
        assert stride >= 1

        super(WideResNetBlock, self).__init__()

        self.normalization = normalization
        """ (bool) Normalization or not. """

        self.dropout = dropout
        """ (bool) Dropout. """

        self.include_bias = include_bias
        """ (bool) Bias. """

        activation = 'relu'

        self.norm1 = get_normalization2d(self.normalization, inplanes)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=self.include_bias)
        #torch.nn.init.xavier_uniform_(self.conv1.weight, gain=numpy.sqrt(2))
        torch.nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity=activation)
        if self.include_bias:
            torch.nn.init.constant_(self.conv1.bias, 0)

        if self.dropout:
            self.drop1 = torch.nn.Dropout(p=0.1)

        self.norm2 = get_normalization2d(self.normalization, planes)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=self.include_bias)
        #torch.nn.init.xavier_uniform_(self.conv2.weight, gain=numpy.sqrt(2))
        torch.nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity=activation)
        if self.include_bias:
            torch.nn.init.constant_(self.conv2.bias, 0)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        if self.dropout:
            out = self.drop1(out)
        out = self.norm2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += self.shortcut(x)

        return out