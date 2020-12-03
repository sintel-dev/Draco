import numpy as np


def slice_by_index(array, axes=None, index=0):
    """Transpose and return the value at the given key."""
    array = np.ndarray.transpose(array, axes)
    return np.ndarray.__getitem__(array, 1)
