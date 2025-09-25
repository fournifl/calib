import numpy as np


def normalize(array, low=0.0, high=255.0):
    """Normalize an array between `min` and `max!

    Input array
    Parameters
    ----------
    array : np.ndarray
        a numpy array, to be normalized
    low : float, int
        minimum value of returned array
    high : float, int
        maximum value of returned array

    Returns:
    -------
    normalized : np.ma.ndarray
        the data from `array`, normalized between `min` and `max`
    """
    old_min = np.nanmin(array)
    old_max = np.nanmax(array)
    factor = 1 if old_min == old_max else high / (old_max - old_min)
    if np.ma.isMaskedArray(array):
        normalized = np.ma.zeros(array.shape)
        normalized.mask = array.mask
        normalized[~array.mask] = (array[~array.mask] - old_min) * factor + low
    else:
        normalized = (array - old_min) * factor + low
    return normalized
