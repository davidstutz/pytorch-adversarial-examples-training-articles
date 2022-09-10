import os
import importlib
import h5py
import numpy


def makedir(dir):
    """
    Creates directory if it does not exist.

    :param dir: directory path
    :type dir: str
    """

    if dir and not os.path.exists(dir):
        os.makedirs(dir)


def get_class(module_name, class_name):
    """
    See https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa.

    :param module_name: module holding class
    :type module_name: str
    :param class_name: class name
    :type class_name: str
    :return: class or False
    """
    # load the module, will raise ImportError if module cannot be loaded
    try:
        m = importlib.import_module(module_name)
    except ImportError as e:
        return False
    # get the class, will raise AttributeError if class cannot be found
    try:
        c = getattr(m, class_name)
    except AttributeError as e:
        return False
    return c


def write_hdf5(filepath, tensors, keys='tensor'):
    """
    Write a simple tensor, i.e. numpy array ,to HDF5.

    :param filepath: path to file to write
    :type filepath: str
    :param tensors: tensor to write
    :type tensors: numpy.ndarray or [numpy.ndarray]
    :param keys: key to use for tensor
    :type keys: str or [str]
    """

    #opened_hdf5() # To be sure as there were some weird opening errors.
    assert type(tensors) == numpy.ndarray or isinstance(tensors, list)
    if isinstance(tensors, list) or isinstance(keys, list):
        assert isinstance(tensors, list) and isinstance(keys, list)
        assert len(tensors) == len(keys)

    if not isinstance(tensors, list):
        tensors = [tensors]
    if not isinstance(keys, list):
        keys = [keys]

    makedir(os.path.dirname(filepath))

    # Problem that during experiments, too many h5df files are open!
    # https://stackoverflow.com/questions/29863342/close-an-open-h5py-data-file
    with h5py.File(filepath, 'w') as h5f:

        for i in range(len(tensors)):
            tensor = tensors[i]
            key = keys[i]

            chunks = list(tensor.shape)
            if len(chunks) > 2:
                chunks[2] = 1
                if len(chunks) > 3:
                    chunks[3] = 1
                    if len(chunks) > 4:
                        chunks[4] = 1

            h5f.create_dataset(key, data=tensor, chunks=tuple(chunks), compression='gzip')
        h5f.close()
        return


def read_hdf5(filepath, key='tensor', efficient=False):
    """
    Read a tensor, i.e. numpy array, from HDF5.

    :param filepath: path to file to read
    :type filepath: str
    :param key: key to read
    :type key: str
    :param efficient: effienct reaidng
    :type efficient: bool
    :return: tensor
    :rtype: numpy.ndarray
    """

    #opened_hdf5() # To be sure as there were some weird opening errors.
    assert os.path.exists(filepath), 'file %s not found' % filepath

    if efficient:
        h5f = h5py.File(filepath, 'r')
        assert key in [key for key in h5f.keys()], 'key %s does not exist in %s with keys %s' % (key, filepath, ', '.join(h5f.keys()))
        return h5f[key]
    else:
        with h5py.File(filepath, 'r') as h5f:
            assert key in [key for key in h5f.keys()], 'key %s does not exist in %s with keys %s' % (key, filepath, ', '.join(h5f.keys()))
            return h5f[key][()]