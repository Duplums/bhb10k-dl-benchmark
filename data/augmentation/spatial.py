from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
from .utils import *


def affine(arr, rotation=10, translation=10, zoom=0.2, order=3, dist="uniform"):
    """ Random affine transformation.

    The affine translation & rotation parameters are drawn from a lognormal
    distribution - small movements are assumed to occur more often and large
    movements less frequently - or from a uniform distribution.

    Parameters
    ----------
    arr: array
        the input data.
    rotation: float or 2-uplet, default 10
        the rotation in degrees of the simulated movements. Larger
        values generate more distorted images.
    translation: float or 2-uplet, default 10
        the translation in voxel of the simulated movements. Larger
        values generate more distorted images.
    zoom: float, default 0.2
        the zooming magnitude. Larger values generate more distorted images.
    order: int, default 3
        the order of the spline interpolation in the range [0, 5].
    dist: str, default 'uniform'
        the sampling distribution: 'uniform' or 'lognormal'.
    Returns
    -------
    transformed: array
        the transformed input data.
    """
    rotation = interval(rotation)
    translation = interval(translation)
    random_rotations = random_generator(
        rotation, arr.ndim, dist=dist)
    random_translations = random_generator(
        translation, arr.ndim, dist=dist)
    random_zooms = random_generator(
        translation, arr.ndim, dist=dist)
    random_zooms = np.random.uniform(
        low=(1 - zoom), high=(1 + zoom), size=arr.ndim)
    random_rotations = Rotation.from_euler(
        "xyz", random_rotations, degrees=True)
    random_rotations = random_rotations.as_dcm()
    affine = compose(random_translations, random_rotations, random_zooms)
    shape = arr.shape
    flow = affine_flow(affine, shape)
    locs = flow.reshape(len(shape), -1)
    transformed = map_coordinates(arr, locs, order=order, cval=0)
    return transformed.reshape(shape)

def flip(arr, axis=None):
    """ Apply a random mirror flip.

    Parameters
    ----------
    arr: array
        the input data.
    axis: int, default None
        apply flip on the specified axis. If not specified, randomize the
        flip axis.

    Returns
    -------
    transformed: array
        the transformed input data.
    """
    if axis is None:
        axis = np.random.randint(low=0, high=arr.ndim, size=1)[0]
    return np.flip(arr, axis=axis)



