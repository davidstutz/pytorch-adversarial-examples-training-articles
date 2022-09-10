import numpy


def uniform_norm(batch_size, dim, epsilon=1, ord=2, low=0, high=1):
    """
    Sample vectors uniformly by norm and direction separately.

    :param batch_size: how many vectors to sample
    :type batch_size: int
    :param dim: dimensionality of vectors
    :type dim: int
    :param epsilon: epsilon-ball
    :type epsilon: float
    :param ord: norm to use
    :type ord: int
    :return: batch_size x dim tensor
    :rtype: numpy.ndarray
    """

    assert ord > 0
    random = numpy.random.randn(batch_size, dim)
    random /= numpy.repeat(numpy.linalg.norm(random, ord=ord, axis=1).reshape(-1, 1), axis=1, repeats=dim)
    random *= epsilon
    uniform = numpy.random.uniform(0, 1, (batch_size, 1))  # exponent is only difference!
    random *= numpy.repeat(uniform, axis=1, repeats=dim)

    return random


def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = numpy.sort(v)[::-1]
    cssv = numpy.cumsum(u) - z
    ind = numpy.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = numpy.maximum(v - theta, 0)
    return w


def project_ball(array, epsilon=1, ord=2):
    """
    Compute the orthogonal projection of the input tensor (as vector) onto the L_ord epsilon-ball.

    **Assumes the first dimension to be batch dimension, which is preserved.**

    :param array: array
    :type array: numpy.ndarray
    :param epsilon: radius of ball.
    :type epsilon: float
    :param ord: order of norm
    :type ord: int
    :return: projected vector
    :rtype: torch.autograd.Variable or torch.Tensor
    """

    assert isinstance(array, numpy.ndarray), 'given tensor should be numpy.ndarray'

    if ord == 0:
        raise NotImplementedError

        # not optimal implementation, see torch variant!
        assert epsilon >= 1
        size = array.shape
        flattened_size = numpy.prod(numpy.array(size[1:]))

        array = array.reshape(-1, flattened_size)
        sorted = numpy.sort(array, axis=1)

        k = int(math.ceil(epsilon))
        thresholds = sorted[:, -k]

        mask = (array >= expand_as(thresholds, array)).astype(float)
        array *= mask
    elif ord == 1:
        size = array.shape
        flattened_size = numpy.prod(numpy.array(size[1:]))

        array = array.reshape(-1, flattened_size)

        for i in range(array.shape[0]):
            # compute the vector of absolute values
            u = numpy.abs(array[i])
            # check if v is already a solution
            if u.sum() <= epsilon:
                # L1-norm is <= s
                continue
            # v is not already a solution: optimum lies on the boundary (norm == s)
            # project *u* on the simplex
            #w = project_simplex(u, s=epsilon)
            w = projection_simplex_sort(u, z=epsilon)
            # compute the solution to the original problem on v
            w *= numpy.sign(array[i])
            array[i] = w

        if len(size) == 4:
            array = array.reshape(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            array = array.reshape(-1, size[1])
    elif ord == 2:
        size = array.shape
        flattened_size = numpy.prod(numpy.array(size[1:]))

        array = array.reshape(-1, flattened_size)
        clamped = numpy.clip(epsilon/numpy.linalg.norm(array, 2, axis=1), a_min=None, a_max=1)
        clamped = clamped.reshape(-1, 1)

        array = array * clamped
        if len(size) == 4:
            array = array.reshape(-1, size[1], size[2], size[3])
        elif len(size) == 2:
            array = array.reshape(-1, size[1])
    elif ord == float('inf'):
        array = numpy.clip(array, a_min=-epsilon, a_max=epsilon)
    else:
        raise NotImplementedError()

    return array