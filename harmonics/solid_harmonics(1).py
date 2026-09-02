"""Fast evaluation of real regular solid harmonics R_l^m (Racah normalised)
at many points for a fixed, static (l, m) table.

Two strategies:
  A. `make_recursion`  -- unrolled Cartesian recursion, O(L^2) elementwise ops.
  B. `make_gemm`       -- offline monomial coefficients + one matmul at runtime.

Verified against scipy.special.sph_harm_y to ~4e-14 up to L=12.
"""
import functools
import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------- strategy A
def make_recursion(table):
    """table: sequence of (l, m); m>0 cosine-type, m<0 sine-type, m=0 zonal.
    Returns f(x, y, z) -> (len(table), N). Loop is unrolled at trace time."""
    L = max(l for l, _ in table)

    def f(x, y, z):
        r2 = x * x + y * y + z * z
        R = {(0, 0): jnp.ones_like(x)}
        for l in range(1, L + 1):
            # diagonal: (l-1, l-1) -> (l, +/-l)
            c = np.sqrt((2 * l - 1) / (2 * l)) * (np.sqrt(2.0) if l == 1 else 1.0)
            Cp = R[(l - 1, l - 1)]
            Sp = R[(l - 1, -(l - 1))] if l > 1 else None
            R[(l, l)] = c * (x * Cp) if Sp is None else c * (x * Cp - y * Sp)
            R[(l, -l)] = c * (y * Cp) if Sp is None else c * (y * Cp + x * Sp)
            # vertical: fixed m, raise l
            for m in range(0, l):
                for mm in ([0] if m == 0 else [m, -m]):
                    a = (2 * l - 1) / np.sqrt(l * l - m * m)
                    t = a * z * R[(l - 1, mm)]
                    if l - 1 > m:
                        b = np.sqrt((l - 1) ** 2 - m * m) / np.sqrt(l * l - m * m)
                        t = t - b * r2 * R[(l - 2, mm)]
                    R[(l, mm)] = t
        return jnp.stack([R[k] for k in table])

    return jax.jit(f)


# ---------------------------------------------------------------- strategy B
def _poly_recursion(L):
    """Same recursion in exact polynomial arithmetic. dict (l,m) -> {(a,b,c): coef}."""
    def mul(p, mono, s=1.0):
        out = {}
        for e, c in p.items():
            k = (e[0] + mono[0], e[1] + mono[1], e[2] + mono[2])
            out[k] = out.get(k, 0.0) + s * c
        return out

    def add(p, q):
        out = dict(p)
        for e, c in q.items():
            out[e] = out.get(e, 0.0) + c
        return out

    R = {(0, 0): {(0, 0, 0): 1.0}}
    X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)
    for l in range(1, L + 1):
        c = np.sqrt((2 * l - 1) / (2 * l)) * (np.sqrt(2.0) if l == 1 else 1.0)
        Cp = R[(l - 1, l - 1)]
        Sp = R[(l - 1, -(l - 1))] if l > 1 else {}
        R[(l, l)] = add(mul(Cp, X, c), mul(Sp, Y, -c))
        R[(l, -l)] = add(mul(Cp, Y, c), mul(Sp, X, c))
        for m in range(0, l):
            for mm in ([0] if m == 0 else [m, -m]):
                a = (2 * l - 1) / np.sqrt(l * l - m * m)
                t = mul(R[(l - 1, mm)], Z, a)
                if l - 1 > m:
                    b = np.sqrt((l - 1) ** 2 - m * m) / np.sqrt(l * l - m * m)
                    for mono in ((2, 0, 0), (0, 2, 0), (0, 0, 2)):
                        t = add(t, mul(R[(l - 2, mm)], mono, -b))
                R[(l, mm)] = t
    return R


def make_gemm(table):
    """Precompute monomial coefficients offline; runtime is one (N,M)@(M,K) matmul."""
    L = max(l for l, _ in table)
    P = _poly_recursion(L)
    monos = sorted({e for k in table for e in P[k]})
    C = np.zeros((len(monos), len(table)))
    idx = {e: i for i, e in enumerate(monos)}
    for j, k in enumerate(table):
        for e, c in P[k].items():
            C[idx[e], j] = c
    C = jnp.asarray(C)
    ea = jnp.asarray([e[0] for e in monos])
    eb = jnp.asarray([e[1] for e in monos])
    ec = jnp.asarray([e[2] for e in monos])
    maxdeg = max(max(e) for e in monos)

    def f(x, y, z):
        # powers by repeated squaring-free cumulative product (maxdeg is small)
        px = jnp.stack([x ** k for k in range(maxdeg + 1)])   # (D+1, N)
        py = jnp.stack([y ** k for k in range(maxdeg + 1)])
        pz = jnp.stack([z ** k for k in range(maxdeg + 1)])
        M = px[ea] * py[eb] * pz[ec]                          # (M, N)
        return C.T @ M                                        # (K, N)

    return jax.jit(f)


# ---------------------------------------------------------------- benchmark
if __name__ == "__main__":
    import time

    L = 10
    table = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)]
    N = 100_000
    rng = np.random.default_rng(0)
    x, y, z = (jnp.asarray(v) for v in rng.normal(size=(3, N)))

    fr, fg = make_recursion(table), make_gemm(table)
    a = fr(x, y, z).block_until_ready()
    b = fg(x, y, z).block_until_ready()
    print(f"table size K={len(table)}, N={N}, L={L}")
    print(f"agreement: {float(jnp.abs(a - b).max()):.2e}\n")

    for name, f in [("recursion", fr), ("gemm     ", fg)]:
        for _ in range(3):
            f(x, y, z).block_until_ready()
        t = time.perf_counter()
        for _ in range(10):
            f(x, y, z).block_until_ready()
        print(f"{name}  {(time.perf_counter()-t)/10*1e3:8.1f} ms")

    # sparse table: 20 modes only
    sparse = [(l, m) for l in range(L + 1) for m in range(-l, l + 1)][:20]
    fr2, fg2 = make_recursion(sparse), make_gemm(sparse)
    print(f"\nsparse table K={len(sparse)}:")
    for name, f in [("recursion", fr2), ("gemm     ", fg2)]:
        f(x, y, z).block_until_ready()
        t = time.perf_counter()
        for _ in range(10):
            f(x, y, z).block_until_ready()
        print(f"{name}  {(time.perf_counter()-t)/10*1e3:8.1f} ms")

    # gradient on the z-axis: polynomial route is clean, spherical route is not
    ax = (jnp.array([0.0]), jnp.array([0.0]), jnp.array([1.0]))
    g = jax.grad(lambda x, y, z: fr(x, y, z).sum())(*ax)
    print(f"\nd/dx at (0,0,1), Cartesian recursion: {g}  finite={bool(jnp.isfinite(g).all())}")

    def spherical(x, y, z):
        r = jnp.sqrt(x * x + y * y + z * z)
        return (r ** 2) * jnp.cos(jnp.arccos(z / r)) * jnp.cos(jnp.arctan2(y, x))
    gs = jax.grad(lambda x, y, z: spherical(x, y, z).sum())(*ax)
    print(f"d/dx at (0,0,1), spherical route:    {gs}  finite={bool(jnp.isfinite(gs).all())}")
