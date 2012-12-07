from astropy.utils.misc import isiterable
from astropy.io import ascii
import astropy.units as u
import numpy as np

__all__ = ["_validate_coord"]

def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])

class SGRData(object):
    _cen_data = ascii.read("data/SGR_CEN", data_start=1, names=["t", "dt", "x", "y", "z", "vx", "vy", "vz"])
    _star_data = None

    def __init__(self, num_stars=1000):
        # Scalings to bring to physical units
        ru = 0.63 # kpc
        vu = (41.27781037*u.km/u.s).to(u.kpc/u.Myr).value # kpc/Myr
        tu = 14.9238134129 # Myr

        if self._star_data == None:
            if num_stars == 0:
                self._star_data = ascii.read("data/SGR_SNAP", data_start=1, names=["m","x","y","z","vx","vy","vz","s1", "s2", "tub"])
            else:
                self._star_data = ascii.read("data/SGR_SNAP", data_start=1, data_end=num_stars+1, names=["m","x","y","z","vx","vy","vz","s1", "s2", "tub"])

        self.sgr_cen = dict()

        self.sgr_cen["x"] = np.array(ru*self._cen_data["x"])
        self.sgr_cen["y"] = np.array(ru*self._cen_data["y"])
        self.sgr_cen["z"] = np.array(ru*self._cen_data["z"])

        self.sgr_cen["vx"] = np.array(vu*self._cen_data["vx"])
        self.sgr_cen["vy"] = np.array(vu*self._cen_data["vy"])
        self.sgr_cen["vz"] = np.array(vu*self._cen_data["vz"])

        self.sgr_cen["t"] = np.array(tu*self._cen_data["t"])
        self.sgr_cen["dt"] = tu*self._cen_data["dt"][0]

        self.sgr_snap = dict()
        self.sgr_snap["x"] = np.array(ru*self._star_data["x"])
        self.sgr_snap["y"] = np.array(ru*self._star_data["y"])
        self.sgr_snap["z"] = np.array(ru*self._star_data["z"])

        self.sgr_snap["vx"] = vu*np.array(self._star_data["vx"])
        self.sgr_snap["vy"] = vu*np.array(self._star_data["vy"])
        self.sgr_snap["vz"] = vu*np.array(self._star_data["vz"])

        self.sgr_snap["tub"] = np.array(tu*self._star_data["tub"])
