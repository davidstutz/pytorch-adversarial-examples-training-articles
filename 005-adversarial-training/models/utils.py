import torch


class ViewOrReshape(torch.nn.Module):
    """
    Simple view layer.
    """

    def __init__(self, *args):
        """
        Constructor.

        :param args: shape
        :type args: [int]
        """

        super(ViewOrReshape, self).__init__()

        self.shape = args

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        if input.is_contiguous():
            return input.view(self.shape)
        else:
            return input.reshape(self.shape)


class Clamp(torch.nn.Module):
    """
    Wrapper for clamp.
    """

    def __init__(self, min=0, max=1):
        """
        Constructor.
        """

        super(Clamp, self).__init__()

        self.min = min
        """ (float) Min value. """

        self.max = max
        """ (float) Max value. """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return torch.clamp(torch.clamp(input, min=self.min), max=self.max)


class Normalize(torch.nn.Module):
    """
    Normalization layer to be learned.
    """

    def __init__(self, n_channels):
        """
        Constructor.

        :param n_channels: number of channels
        :type n_channels: int
        """

        super(Normalize, self).__init__()

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = torch.nn.Parameter(torch.ones(n_channels))
        self.bias = torch.nn.Parameter(torch.zeros(n_channels))
        # buffers are not saved in state dict!
        #self.register_buffer('std', torch.ones(n_channels))
        #self.register_buffer('mean', torch.zeros(n_channels))

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return (input - self.bias.view(1, -1, 1, 1))/self.weight.view(1, -1, 1, 1)


class Flatten(torch.nn.Module):
    """
    Flatten vector, allows to flatten without knowing batch_size and flattening size.
    """

    def forward(self, input):
        """
        Forward pass.

        :param input: input
        :type input: torch.autograd.Variable
        :return: output
        :rtype: torch.autograd.Variable
        """

        return input.view(input.size(0), -1)


def get_normalization2d(normalization, planes):
    assert normalization in [
        '',
        'bn',
        'fixedbn',
        'gn',
        'fixedgn',
    ]

    num_group_alternatives = [32, 24, 16]
    for i in range(len(num_group_alternatives)):
        num_groups = min(num_group_alternatives[i], planes // 2)
        if planes % num_groups == 0:
            break
    assert planes % num_groups == 0

    norm = torch.nn.Identity()
    if normalization == 'bn':
        norm = torch.nn.BatchNorm2d(planes)
        torch.nn.init.constant_(norm.weight, 1)
        torch.nn.init.constant_(norm.bias, 0)

    elif normalization == 'fixedbn':
        norm = torch.nn.BatchNorm2d(planes, affine=False)

    elif normalization == 'gn':
        norm = torch.nn.GroupNorm(num_groups, planes)
        torch.nn.init.constant_(norm.weight, 1)
        torch.nn.init.constant_(norm.bias, 0)

    elif normalization == 'fixedgn':
        norm = torch.nn.GroupNorm(num_groups, planes, affine=False)

    assert isinstance(norm, torch.nn.Identity) or norm != ''
    return norm


def get_normalization1d(normalization, out_features):
    assert normalization in [
        '',
        'bn',
        'fixedbn',
    ]

    norm = torch.nn.Identity()
    if normalization == 'bn':
        norm = torch.nn.BatchNorm1d(out_features)
        torch.nn.init.constant_(norm.weight, 1)
        torch.nn.init.constant_(norm.bias, 0)

    elif normalization == 'fixedbn':
        norm = torch.nn.BatchNorm1d(out_features, affine=False)

    assert isinstance(norm, torch.nn.Identity) or norm != ''
    return norm