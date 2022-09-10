import os
import importlib
import functools


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


def partial(f, *args, **kwargs):
    """
    Create partial while preserving __name__ and __doc__.

    :param f: function
    :type f: callable
    :param args: arguments
    :type args: dict
    :param kwargs: keyword arguments
    :type kwargs: dict
    :return: partial
    :rtype: callable
    """
    p = functools.partial(f, *args, **kwargs)
    functools.update_wrapper(p, f)
    return p