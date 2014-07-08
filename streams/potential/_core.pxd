
cdef class Potential:
    cpdef evaluate(self, double[:,::1] r)
    cdef public void _evaluate(self, double[:,::1] r, double[::1] pot, int nparticles)
    cpdef acceleration(self, double[:,::1] r)
    cdef public void _acceleration(self, double[:,::1] r, double[:,::1] acc, int nparticles)