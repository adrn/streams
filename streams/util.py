from astropy.utils.misc import isiterable
import numpy as np

__all__ = ["_validate_coord", "cartesian3D", "cylindrical3D", "spherical3D", "CoordinateSystem"]

def _validate_coord(x):
    if isiterable(x):
        return np.array(x, copy=True)
    else:
        return np.array([x])

class frozendict(dict):
    def _blocked_attribute(obj):
        raise AttributeError, "A frozendict cannot be modified."
    _blocked_attribute = property(_blocked_attribute)

    __delitem__ = __setitem__ = clear = _blocked_attribute
    pop = popitem = setdefault = update = _blocked_attribute

    def __new__(cls, *args, **kw):
        new = dict.__new__(cls)

        args_ = []
        for arg in args:
            if isinstance(arg, dict):
                arg = copy.copy(arg)
                for k, v in arg.items():
                    if isinstance(v, dict):
                        arg[k] = frozendict(v)
                    elif isinstance(v, list):
                        v_ = list()
                        for elm in v:
                            if isinstance(elm, dict):
                                v_.append( frozendict(elm) )
                            else:
                                v_.append( elm )
                        arg[k] = tuple(v_)
                args_.append( arg )
            else:
                args_.append( arg )

        dict.__init__(new, *args_, **kw)
        return new

    def __init__(self, *args, **kw):
        pass

    def __hash__(self):
        try:
            return self._cached_hash
        except AttributeError:
            h = self._cached_hash = hash(tuple(sorted(self.items())))
            return h

    def __repr__(self):
        return "frozendict(%s)" % dict.__repr__(self)

class CoordinateSystem(object):

    def __init__(self, *args):
        """ Register a coordinate system. """

        self.ndim = len(args)
        self.axis_names = [str(arg) for arg in args]

cartesian3D = CoordinateSystem("x", "y", "z")
cylindrical3D = CoordinateSystem("R", "phi", "z")
spherical3D = CoordinateSystem("r", "phi", "theta")