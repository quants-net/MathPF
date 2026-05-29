"""Even vs odd CF convergent orders for -R'''.
For each order n, evaluate the convergent -R'''^[n] via the P/Q recurrence
(P_{m+1}=x P_m + m P_{m-1}, P0=1,P1=x; Q0=0,Q1=1), compare to the true -R''',
and bisect for x_cf (smallest x where relative error <= eps).
Cost = x^2-Horner depth of the denominator P_{n+1} = floor((n+1)/2)."""
import mpmath as mp
mp.mp.dps = 80
EPS = 2.0 ** -52


def PQ(n1, x):
    """Return P_{n1}(x), Q_{n1}(x)."""
    P0, P1 = mp.mpf(1), x
    Q0, Q1 = mp.mpf(0), mp.mpf(1)
    for m in range(1, n1):
        P0, P1 = P1, x * P1 + m * P0
        Q0, Q1 = Q1, x * Q1 + m * Q0
    return P1, Q1


def conv_Rd3(n, x):
    P, Q = PQ(n + 1, x)
    mRp = (P - x * Q) / P          # -R'^[n]
    return (x * x + 3) * mRp - 1   # -R'''^[n]


def true_Rd3(x):
    R = mp.sqrt(mp.pi / 2) * mp.erfc(x / mp.sqrt(2)) * mp.e ** (x * x / 2)
    return (x * x + 3) * (1 - x * R) - 1


def xcf(n):
    lo, hi = mp.mpf('1'), mp.mpf('1e9')
    relerr = lambda x: abs((conv_Rd3(n, x) - true_Rd3(x)) / true_Rd3(x))
    for _ in range(200):
        mid = mp.sqrt(lo * hi)
        if relerr(mid) <= EPS:
            hi = mid
        else:
            lo = mid
    return hi


print(f"{'n':>3} {'parity':>6} {'cost k':>7} {'x_cf (rel err = eps)':>22}")
for n in range(3, 11):
    k = (n + 1) // 2
    par = "even" if n % 2 == 0 else "ODD"
    print(f"{n:>3} {par:>6} {k:>7} {float(xcf(n)):>22.4g}")
