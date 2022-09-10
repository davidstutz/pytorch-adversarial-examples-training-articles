import torch
import common.torch


class STNDecoder(torch.nn.Module):
    """
    Combine decoder and classifier in single model for attacks.
    """

    def __init__(self, N_theta, interpolation_mode='bilinear', padding_mode='zeros'):
        """
        Constructor.

        :param N_theta: number of transformations
        :type N_theta: int
        """

        super(STNDecoder, self).__init__()

        self.images = None
        """ (torch.autograd.Variable) Images. """

        assert N_theta > 0 and N_theta <= 6
        self.N_theta = N_theta
        """ (int) Number of transformations. """

        assert interpolation_mode in ['bilinear', 'nearest']
        self.interpolation_mode = interpolation_mode

        assert padding_mode in ['zeros', 'border', 'reflection']
        self.padding_mode = padding_mode

    def set_images(self, images):
        """
        Set images for STN.

        :param images: images
        :type images: torch.autograd.Variable
        """

        self.images = images

    def forward(self, theta):
        """
        Forward pass through both models.

        :param theta: latent codes
        :type theta: torch.autograd.Variable
        :return: classification
        :rtype: torch.autograd.Variable
        """

        assert self.images is not None

        if self.N_theta > 0:
            translation_x = theta[:, 0]
        if self.N_theta > 1:
            translation_y = theta[:, 1]
        if self.N_theta > 2:
            shear_x = theta[:, 2]
        if self.N_theta > 3:
            shear_y = theta[:, 3]
        if self.N_theta > 4:
            scales = 1 + theta[:, 4]
        if self.N_theta > 5:
            rotation = theta[:, 5]

        transformation = torch.autograd.Variable(torch.FloatTensor(theta.size()[0], 6).fill_(0))
        if common.torch.is_cuda(theta):
            transformation = transformation.cuda()

        # translation x
        if self.N_theta == 1:
            transformation[:, 0] = 1
            transformation[:, 4] = 1
            transformation[:, 2] = translation_x
        # translation y
        elif self.N_theta == 2:
            transformation[:, 0] = 1
            transformation[:, 4] = 1
            transformation[:, 2] = translation_x
            transformation[:, 5] = translation_y
        # shear x
        elif self.N_theta == 3:
            transformation[:, 0] = 1
            transformation[:, 1] = shear_x
            transformation[:, 2] = translation_x
            transformation[:, 4] = 1
            transformation[:, 5] = translation_y
        # shear y
        elif self.N_theta == 4:
            transformation[:, 0] = 1
            transformation[:, 1] = shear_x
            transformation[:, 2] = translation_x
            transformation[:, 3] = shear_y
            transformation[:, 4] = 1
            transformation[:, 5] = translation_y
        # scale
        elif self.N_theta == 5:
            transformation[:, 0] = scales
            transformation[:, 1] = scales * shear_x
            transformation[:, 2] = translation_x
            transformation[:, 3] = scales * shear_y
            transformation[:, 4] = scales
            transformation[:, 5] = translation_y
        # rotation
        elif self.N_theta == 6:
            transformation[:, 0] = torch.cos(rotation) * scales - torch.sin(rotation) * scales * shear_x
            transformation[:, 1] = -torch.sin(rotation) * scales + torch.cos(rotation) * scales * shear_x
            transformation[:, 2] = translation_x
            transformation[:, 3] = torch.cos(rotation) * scales * shear_y + torch.sin(rotation) * scales
            transformation[:, 4] = -torch.sin(rotation) * scales * shear_y + torch.cos(rotation) * scales
            transformation[:, 5] = translation_y
        else:
            raise NotImplementedError()

        transformation = transformation.view(-1, 2, 3)
        grid = torch.nn.functional.affine_grid(transformation, self.images.size())
        output = torch.nn.functional.grid_sample(self.images, grid, mode=self.interpolation_mode, padding_mode=self.padding_mode)
        output = torch.clamp(torch.clamp(output, min=0), max=1)
        return output

    def eval(self):
        """
        Just for saying that eval needs to be called on encoder and decoder explicitly.
        """

        raise NotImplementedError('eval needs to be called on decoder and classifier separately!')