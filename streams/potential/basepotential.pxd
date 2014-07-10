
cdef class _Potential:
    cpdef evaluate(self, double[:,::1] xyz)
    cdef public void _evaluate(self, double[:,::1] xyz, double[::1] pot, int nparticles)

    cpdef acceleration(self, double[:,::1] xyz)
    cdef public void _acceleration(self, double[:,::1] xyz, double[:,::1] acc, int nparticles)

    cpdef var_acceleration(self, double[:,::1] w)
    cdef public void _var_acceleration(self, double[:,::1] w, double[:,::1] acc, int nparticles)

    cpdef tidal_radius(self, double m_sat, double[::1] r)
    cdef public double _tidal_radius(self, double m_sat, double R)