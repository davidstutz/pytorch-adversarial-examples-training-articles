import random
import numpy as np

""" 
Stores the four directions in the form of the change in pixel values along each dimension of
the image
"""
MaskDirection = {
    'left': (0, -1),
    'right': (0, 1),
    'up': (-1, 0),
    'down': (1, 0)
}


# for Python < 3.6
def choices(sequence, k):
    return [random.choice(sequence) for _ in range(k)]


class MaskGenerator:
    """
    Generates a frame location.
    """

    def __init__(self, img_dims):
        """
        Constructor.

        :param img_dims: image dimensions
        :type img_dims: tuple
        """
        assert len(img_dims) == 2

        self.img_dims = img_dims

    def random_location(self, n=1):
        """
        Generates n mask coordinates randomly from allowed locations

        :param n: number of masks to generate, defaults to 1
        :type n: int, optional
        :return: n masks
        :rtype: numpy.array
        """

        raise NotImplementedError

    def get_masks(self, mask_coords, n_channels):
        """
        Gets mask in image shape given mask coordinates

        :param mask_coords: mask coordinates for the batch of masks
        :type mask_coords: numpy.array
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """

        raise NotImplementedError


class FrameGenerator(MaskGenerator):
    """
    Generates a frame location.
    """

    def __init__(self, img_dims, frame_size):
        """
        Constructor.

        :param img_dims: image dimensions
        :type img_dims: tuple
        :param frame_size: frame size in pixels
        :type frame_size: int
        """

        super(FrameGenerator, self).__init__(img_dims)

        self.frame_size = frame_size

    def random_location(self, n=1):
        """
        Generates n mask coordinates randomly from allowed locations

        :param n: number of masks to generate, defaults to 1
        :type n: int, optional
        :return: n masks
        :rtype: numpy.array
        """

        return [[0, 0] for _ in range(n)]

    def get_masks(self, mask_coords, n_channels):
        """
        Gets mask in image shape given mask coordinates

        :param mask_coords: mask coordinates for the batch of masks
        :type mask_coords: numpy.array
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """
        assert n_channels >= 1

        batch_size = len(mask_coords)
        masks = np.zeros((batch_size, n_channels, self.img_dims[0], self.img_dims[1]))
        masks[:, :, :, :self.frame_size] = 1
        masks[:, :, :self.frame_size, :] = 1
        masks[:, :, :, -self.frame_size:] = 1
        masks[:, :, -self.frame_size:, :] = 1

        return masks


class PatchGenerator(MaskGenerator):
    """
    Generates a random mask location for applying an adversarial patch to an image
    """

    def __init__(self, img_dims, mask_dims, exclude_list=None):
        """
        Constructor.

        :param img_dims: image dimensions
        :type img_dims: tuple
        :param mask_dims: mask dimensions
        :type mask_dims: tuple
        :param include_list: list of boxes in image to include among possible mask locations, defaults to None
        :type include_list: numpy.array, optional
        """

        super(PatchGenerator, self).__init__(img_dims)

        assert len(mask_dims) == 2
        assert mask_dims <= img_dims

        self.mask_dims = mask_dims
        self.exclude_list = exclude_list
        if self.exclude_list is not None:
            self.allowed_pixels = self._parse_exclude_list()
        else:
            self.allowed_pixels = set()
            y = 0
            x = 0
            h = img_dims[0] - 1
            w = img_dims[1] - 1
            assert x >= 0 and y >= 0 and h > 0 and w > 0
            assert y + h < self.img_dims[0] and x + w < self.img_dims[1]
            y_range = np.arange(y, y + h - self.mask_dims[0] + 1)
            x_range = np.arange(x, x + w - self.mask_dims[1] + 1)
            pixels = [(y, x) for y in y_range for x in x_range]
            self.allowed_pixels.update(pixels)

    def _parse_exclude_list(self):
        """
        Generates list of disallowed pixels to start mask

        :return: list of disallowed pixels
        :rtype: set
        """
        assert len(self.exclude_list.shape) == 2
        all_pixels = [(y, x) for y in range(self.img_dims[0]-self.mask_dims[0])
                      for x in range(self.img_dims[1]-self.mask_dims[1])]
        allowed_pixels = set(all_pixels)
        for box in self.exclude_list:
            y, x, h, w = box
            assert x >= 0 and y >= 0 and h > 0 and w > 0
            assert y+h < self.img_dims[0] and x+w < self.img_dims[1]
            y_range = np.arange(max(0, y-self.mask_dims[0]+1), y+h)
            x_range = np.arange(max(0, x-self.mask_dims[1]+1), x+w)
            pixels = [(y, x) for y in y_range for x in x_range]
            allowed_pixels = allowed_pixels.difference(pixels)
        return allowed_pixels

    def random_location(self, n=1):
        """
        Generates n mask coordinates randomly from allowed locations

        :param n: number of masks to generate, defaults to 1
        :type n: int, optional
        :return: n masks
        :rtype: numpy.array
        """
        assert n >= 1
        assert len(self.allowed_pixels) > 0
        start_pixels = choices(tuple(self.allowed_pixels), k=n)
        return np.array([(y, x, self.mask_dims[0], self.mask_dims[1]) for (y, x) in start_pixels])

    def random_batch(self, batch_size, n_channels):
        """
        Generates random masks of specified shape

        :param batch_size: number of masks to generate
        :type batch_size: int
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """
        assert batch_size >= 1 and n_channels >= 1
        assert len(self.allowed_pixels) > 0
        mask_coords = choices(tuple(self.allowed_pixels), k=batch_size)
        masks = np.zeros((batch_size, n_channels, self.img_dims[0], self.img_dims[1]))
        for b in range(batch_size):
            masks[b, :, mask_coords[b][0]:mask_coords[b][0]+self.mask_dims[0],
                  mask_coords[b][1]:mask_coords[b][1]+self.mask_dims[1]] = 1
        return masks

    def get_masks(self, mask_coords, n_channels):
        """
        Gets mask in image shape given mask coordinates

        :param mask_coords: mask coordinates for the batch of masks
        :type mask_coords: numpy.array
        :param n_channels: number of channels in image, used as second dimension of masks
        :type n_channels: int
        :return: generated masks
        :rtype: numpy.array
        """
        assert n_channels >= 1
        batch_size = len(mask_coords)
        masks = np.zeros((batch_size, n_channels, self.img_dims[0], self.img_dims[1]))
        for b in range(batch_size):
            masks[b, :, mask_coords[b][0]:mask_coords[b][0]+self.mask_dims[0],
                  mask_coords[b][1]:mask_coords[b][1]+self.mask_dims[1]] = 1
        return masks

    def move_coords(self, mask_coords, direction, stride=1):
        """
        Moves mask coordinates in a specified direction using a specified stride. Each coordinate
        is moved only if the new location is an allowed location.

        :param mask_coords: mask coordinates to move
        :type mask_coords: numpy.array
        :param direction: direction to move mask coordinate, key in MaskDirection dict
        :type direction: str
        :param stride: stride when moving coordinate, defaults to 1
        :type stride: int, optional
        :return: moved coordinates
        :rtype: numpy.array
        """
        batch_size = len(mask_coords)
        new_coords = np.copy(mask_coords)
        for b in range(batch_size):
            new_coords[b] = self.move_coords_single(mask_coords[b], direction, stride)
        return new_coords
