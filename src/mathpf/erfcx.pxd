# Cython header for mathpf.erfcx -- cimport the scalar kernels, e.g.
#   from mathpf.erfcx cimport _erfcx, _erfcx_d1, _erfcx_d3
#
# The kernels themselves are implemented in C++ (src/mathpf/_kernels/erfcx.cpp,
# templated on T in {float, double}, namespace mathpf) and exposed through the
# stable extern "C" double-only ABI declared in _kernels/erfcx.h.  Cython
# names (_erfcx, _erfcx_d1, _erfcx_d3) are name-aliased to the mathpf_*
# C symbols so any downstream Cython module can cimport unchanged.
cdef extern from "_kernels/erfcx.h":
    double _erfcx    "mathpf_Erfcx"        (double z) noexcept nogil
    double _erfcx_d1 "mathpf_ErfcxDeriv1"  (double z) noexcept nogil
    double _erfcx_d3 "mathpf_ErfcxDeriv3"  (double z) noexcept nogil
