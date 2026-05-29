"""Reference pure-Python implementations of mathpf's primitives.

These modules are the canonical Python specification of what the compiled
top-level surface (millsratio, millsratio_d1, millsratio_d3, millsratio_dd,
millsratio_dd_cf, etc.) computes.  They are used by mathpf's test suite as a
bit-equality reference, and are shipped with the wheel as readable source for
anyone studying or porting the algorithms.

Production code should call the top-level mathpf functions for performance.
The _pyref implementations are scalar Python and are several orders of
magnitude slower than the compiled binding.
"""
