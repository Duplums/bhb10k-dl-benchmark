from models.densenet import DenseNet


def tiny_densenet121(memory_efficient=False, **kwargs):
    r"""Tiny-Densenet-121 model from
     "Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>

    The tiny-version has been specifically designed for neuroimaging data. It is 10X smaller than DenseNet.
    Args:
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return DenseNet(16, (6, 12, 16), 64, memory_efficient=memory_efficient, **kwargs)
