import math
import matplotlib
from matplotlib import pyplot
import numpy
import skimage.transform
import skimage


# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# https://matplotlib.org/users/usetex.html
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
# https://matplotlib.org/users/customizing.html
matplotlib.rcParams['lines.linewidth'] = 1
#matplotlib.rcParams['figure.figsize'] = (30, 10)


def mosaic(images, scale=1, cols=6, cmap='binary', vmin=None, vmax=None, ax=pyplot.gca(), **kwargs):
    """
    Create a mosaic of images, specifically display a set of images in
    a variable number of rows with a fixed number of columns.

    :param images: images
    :type images: numpy.ndarray
    :param cols: number of columns
    :type cols: int
    :param cmap: color map to use, default is binary
    :type cmap: str
    """

    assert len(images.shape) == 4, 'set of images expected, rank 3 or 4 required'
    assert images.shape[0] > 0

    number = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    channels = images.shape[3]

    rows = int(math.ceil(number / float(cols)))
    image = numpy.zeros((rows * height, cols * width, channels))

    for i in range(rows):
        for j in range(cols):
            k = i * cols + j
            if k < images.shape[0]:
                image[i * height: (i + 1) * height, j * width: (j + 1) * width, :] = images[k, :, :, :]

    image = numpy.squeeze(image)
    image = skimage.transform.rescale(image, scale=scale, order=0, multichannel=len(images.shape) == 3)
    ax.imshow(image, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
