# Cython header for mathpf.mills -- cimport the scalar kernels, e.g.
#   from mathpf.mills cimport _R, _R1, _R3, _R013_CF, _Rrel_below1
#
# The kernels themselves are implemented in C++ (src/mathpf/_kernels/mills.cpp,
# templated on T in {float, double}, namespace QuantsNet) and exposed through
# the stable extern "C" double-only ABI declared in _kernels/mills.h.  Cython
# names (_R, _R1, etc.) are name-aliased to the mathpf_* C symbols so that any
# downstream Cython module continues to `cimport _R` unchanged.
cdef extern from "_kernels/mills.h":
    double _R           "mathpf_MillsRatio"            (double x) noexcept nogil
    double _R1          "mathpf_MillsRatioDeriv1"      (double x) noexcept nogil
    double _R3          "mathpf_MillsRatioDeriv3"      (double x) noexcept nogil
    double _Rrel_below1 "mathpf_MillsRatioRel_below1"  (double x) noexcept nogil
    double _R013_CF     "mathpf_MillsRatio_CF"         (double x_or_u, int n, int d) noexcept nogil
