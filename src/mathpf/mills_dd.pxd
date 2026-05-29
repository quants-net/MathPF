# Cython header for mathpf.mills_dd -- cimport the scalar kernels, e.g.
#   from mathpf.mills_dd cimport _R_DD, _R_DD_CF
#
# The kernels themselves are implemented in C++ (src/mathpf/_kernels/mills_dd.cpp,
# templated on T in {float, double}, namespace QuantsNet), exposed through the
# stable extern "C" double-only ABI declared in _kernels/mills_dd.h.  The Cython
# names are aliased to the mathpf_* C symbols so any downstream Cython module can
# `cimport _R_DD` / `cimport _R_DD_CF` unchanged.
cdef extern from "_kernels/mills_dd.h":
    double _R_DD    "mathpf_MillsRatioDiff"     (double x, double dx, int theta)   noexcept nogil
    double _R_DD_CF "mathpf_MillsRatioDiff_CF"  (double x, double dx, int n_terms) noexcept nogil
