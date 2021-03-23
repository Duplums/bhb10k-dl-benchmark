from scipy.spatial.transform import Rotation
from scipy.ndimage import map_coordinates
from skimage import transform
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


def crop(arr, shape, crop_type="center", resize=False, keep_dim=False):
    """Crop the given n-dimensional array either at a random location or centered
    :param
            shape: tuple or list of int
                The shape of the patch to crop
            crop_type: 'center' or 'random'
                Wheter the crop will be centered or at a random location
            resize: bool, default False
                If True, resize the cropped patch to the inital dim. If False, depends on keep_dim
            keep_dim: bool, default False
                if True and resize==False, put a constant value around the patch cropped. If resize==True, does nothing
    """
    assert isinstance(arr, np.ndarray)
    assert type(shape) == int or len(shape) == len(arr.shape), "Shape of array {} does not match {}". \
        format(arr.shape, shape)

    img_shape = np.array(arr.shape)
    if type(shape) == int:
        size = [shape for _ in range(len(shape))]
    else:
        size = np.copy(shape)
    indexes = []
    for ndim in range(len(img_shape)):
        if size[ndim] > img_shape[ndim] or size[ndim] < 0:
            size[ndim] = img_shape[ndim]
        if crop_type == "center":
            delta_before = int((img_shape[ndim] - size[ndim]) / 2.0)
        elif crop_type == "random":
            delta_before = np.random.randint(0, img_shape[ndim] - size[ndim] + 1)
        indexes.append(slice(delta_before, delta_before + size[ndim]))
    if resize:
        # resize the image to the input shape
        return transform.resize(arr[tuple(indexes)], img_shape, preserve_range=True)

    if keep_dim:
        mask = np.zeros(img_shape, dtype=np.bool)
        mask[tuple(indexes)] = True
        arr_copy = arr.copy()
        arr_copy[~mask] = 0
        return arr_copy

    return arr[tuple(indexes)]


def padding(arr, shape, **kwargs):
        """Fill an array to fit the desired shape.
        :param
        arr: np.array
            an input array.
        **kwargs: params to give to np.pad (value to fill, etc.)
        :return
        fill_arr: np.array
            the padded array.
        """
        orig_shape = arr.shape
        padding = []
        for orig_i, final_i in zip(orig_shape, shape):
            shape_i = final_i - orig_i
            half_shape_i = shape_i // 2
            if shape_i % 2 == 0:
                padding.append([half_shape_i, half_shape_i])
            else:
                padding.append([half_shape_i, half_shape_i + 1])
        for cnt in range(len(arr.shape) - len(padding)):
            padding.append([0, 0])
        fill_arr = np.pad(arr, padding, **kwargs)
        return fill_arr
