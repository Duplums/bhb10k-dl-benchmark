import numbers
import math
import numpy as np


def interval(obj, lower=None):
    """ Listify an object.

    Parameters
    ----------
    obj: 2-uplet or number
        the object used to build the interval.
    lower: number, default None
        the lower bound of the interval. If not specified, a symetric
        interval is generated.

    Returns
    -------
    interval: 2-uplet
        an interval.
    """
    if isinstance(obj, numbers.Number):
        if obj < 0:
            raise ValueError("Specified interval value must be positive.")
        if lower is None:
            lower = -obj
        return (lower, obj)
    if len(obj) != 2:
        raise ValueError("Interval must be specified with 2 values.")
    min_val, max_val = obj
    if min_val > max_val:
        raise ValueError("Wrong interval boudaries.")
    return tuple(obj)

def random_generator(interval, size, dist="uniform"):
    """ Random varaible generator.

    Parameters
    ----------
    interval: 2-uplet
        the possible values of the generated random variable.
    size: uplet
        the number of random variables to be drawn from the sampling
        distribution.
    dist: str, default 'uniform'
        the sampling distribution: 'uniform' or 'lognormal'.

    Returns
    -------
    random_variables: array
        the generated random variable.
    """
    np.random.seed()
    if dist == "uniform":
        random_variables = np.random.uniform(
            low=interval[0], high=interval[1], size=size)
    # max height occurs at x = exp(mean - sigma**2)
    # FWHM is found by finding the values of x at 1/2 the max height =
    # exp((mean - sigma**2) + sqrt(2*sigma**2*ln(2))) - exp((mean - sigma**2)
    # - sqrt(2*sigma**2*ln(2)))
    elif dist == "lognormal":
        sign = np.random.randint(0, 2, size=size) * 2 - 1
        sign = sign.astype(np.float)

        random_variables = np.random.lognormal(mean=0., sigma=1., size=size)
        random_variables /= 12.5
        random_variables *= (sign * interval[1])
    else:
        raise ValueError("Unsupported sampling distribution.")
    return random_variables

def affine_flow(affine, shape):
    """ Generates a flow field given an affine matrix.

    Parameters
    ----------
    affine: Tensor (4, 4)
        an affine transform.
    shape: tuple
        the target output image shape.

    Returns
    -------
    flow: Tensor
        the generated affine flow field.
    """
    ranges = [np.arange(size) for size in shape]
    ranges[0], ranges[1] = ranges[1], ranges[0]
    mesh = np.asarray(np.meshgrid(*ranges))
    mesh[[0, 1]] = mesh[[1, 0]]
    locs = np.asarray(mesh)
    offset = (np.array(shape) - 1) / 2
    locs = np.transpose(locs.T - offset)
    ones = np.ones([1] + list(locs.shape)[1:], dtype=locs.dtype)
    homography_grid = np.concatenate([locs, ones], axis=0)
    shape = homography_grid.shape
    homography_grid = homography_grid.reshape(shape[0], -1)
    flow = np.dot(affine, homography_grid)
    flow = flow.reshape(shape)
    flow = flow[:(len(shape) - 1)]
    flow = np.transpose(flow.T + offset)
    return flow


def compose(T, R, Z, S=None):
    """ Compose translations, rotations, zooms, [shears]  to affine

    Parameters
    ----------
    T: array (N,)
        translations, where N is usually 3 (3D case)
    R: array (N, N)
        rotation matrix where N is usually 3 (3D case)
    Z: array (N,)
        zooms, where N is usually 3 (3D case)
    S: array (P,), default None
        shear vector, such that shears fill upper triangle above
        diagonal to form shear matrix.  P is the (N-2)th Triangular
        number, which happens to be 3 for a 4x4 affine (3D case)

    Returns
    -------
    A: array (N+1, N+1)
        affine transformation matrix where N usually == 3
        (3D case)
    """
    n = len(T)
    R = np.asarray(R)
    if R.shape != (n, n):
        raise ValueError("Wrong rotation matrix shape.")
    A = np.eye(n + 1)
    if S is not None:
        Smat = striu2mat(S)
        ZS = np.dot(np.diag(Z), Smat)
    else:
        ZS = np.diag(Z)
    A[:n, :n] = np.dot(R, ZS)
    A[:n, n] = T[:]
    return A


# Caching dictionary for common shear Ns, indices
_shearers = {}
for n in range(1, 11):
    x = (n**2 + n) / 2.0
    i = n + 1
    _shearers[x] = (i, np.triu(np.ones((i, i)), 1).astype(bool))


def striu2mat(striu):
    """ Construct shear matrix from upper triangular vector.

    Parameters
    ----------
    striu: array (N,)
       vector giving triangle above diagonal of shear matrix.

    Returns
    -------
    SM: array (N, N)
       shear matrix.
    """
    n = len(striu)
    if n in _shearers:
        N, inds = _shearers[n]
    else:
        N = ((-1 + math.sqrt(8 * n + 1)) / 2.0) + 1  # n+1 th root
        if N != math.floor(N):
            raise ValueError(
                "{0} is a strange number of shear elements".format(n))
        N = int(N)
        inds = np.triu(np.ones((N, N)), 1).astype(bool)
    M = np.eye(N)
    M[inds] = striu
    return M
