# Cython header for mathpf.mills_dd -- cimport the scalar kernels, e.g.
#   from mathpf.mills_dd cimport _R_DD, _R_DD_asymp_x2
cdef double _R_DD(double x, double dx, int theta) noexcept nogil
cdef double _R_DD_asymp_x2(double x, double dx, int n_terms) noexcept nogil
