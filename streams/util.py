from astropy.utils.misc import isiterable
import numpy as np

def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])