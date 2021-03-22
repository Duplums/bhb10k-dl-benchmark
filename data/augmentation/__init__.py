# -*- coding: utf-8 -*-
##########################################################################
# NSAp - Copyright (C) CEA, 2020
# Distributed under the terms of the CeCILL-B license, as published by
# the CEA-CNRS-INRIA. Refer to the LICENSE file or to
# http://www.cecill.info/licences/Licence_CeCILL-B_V1-en.html
# for details.
##########################################################################

"""
Module that privides common spatial and intensity data augmentation tools.
"""

# Import
from collections import namedtuple
import copy
import logging
import numpy as np
from .spatial import affine
from .spatial import flip
from .spatial import deformation
from .spatial import cutout
from .intensity import add_blur
from .intensity import add_noise
from .intensity import add_ghosting
from .intensity import add_spike
from .intensity import add_biasfield
from .intensity import add_motion
from .intensity import add_offset
from .intensity import add_swap

# Global parameters
logger = logging.getLogger("pynet")

class Transformer(object):
    """ Class that can be used to register a sequence of transformations.
    """
    Transform = namedtuple("Transform", ["transform", "params", "probability",
                                         "apply_to", "with_channel"])

    def __init__(self, with_channel=True, output_label=False):
        """ Initialize the class.

        Parameters
        ----------
        with_channel: bool, default True
            the input array shape to be transformd is (C, *), where C
            represents the channel dimension. To omit the channel dimension
            unset this parameter.
        output_label: bool, default False
            if output data are labels, automatically force the interpolation
            to nearest neighboor via the 'order' transform parameter.
        """
        self.transforms = []
        self.dtype = "all"
        self.with_channel = with_channel
        self.output_label = output_label


    def register(self, transform, probability=1, apply_to=None, with_channel=False, **kwargs):
        """ Register a new transformation.

        Parameters
        ----------
        transform: callable
            the transformation function.
        probability: float, default 1
            the transform is applied with the specified probability.
        apply_to: list of str, default None
            the registered transform will be only applied on specified
            data - 'all', 'input' or 'output'.
        kwargs
            the transformation function parameters.
        """
        if apply_to is None:
            apply_to = ["all"]
        trf = self.Transform(
            transform=transform, params=kwargs, probability=probability,
            apply_to=apply_to, with_channel=with_channel)
        self.transforms.append(trf)

    def __call__(self, arr):
        """ Apply the registered transformations.

        Parameters
        ----------
        arr: array
            the input data.

        Returns
        -------
        transformed: array
            the transformed input data.
        """
        transformed = arr.copy()
        if not self.with_channel:
            transformed = np.expand_dims(transformed, axis=0)
        for trf in self.transforms:
            if self.dtype not in trf.apply_to:
                continue
            kwargs = copy.deepcopy(trf.params)
            if (self.output_label and self.dtype == "output" and
                    "order" in kwargs):
                kwargs["order"] = 0
            if np.random.rand() < trf.probability:
                logger.debug("Applying {0}...".format(trf.transform))
                if trf.with_channel:
                    transformed = trf.transform(transformed, **kwargs)
                else:
                    for channel_id in range(transformed.shape[0]):
                        transformed[channel_id] = trf.transform(
                            transformed[channel_id], **kwargs)
                logger.debug("Done.")
        if not self.with_channel:
            transformed = transformed[0]
        return transformed

    def __str__(self):
        if len(self.transforms) == 0:
            return '(Empty transformer)'
        s = 'Composition of:'
        for trf in self.transforms:
            s += '\n\t- '+trf.__str__()
        return s