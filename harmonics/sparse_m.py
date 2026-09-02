import numpy as np, jax, jax.numpy as jnp, time, functools
jax.config.update("jax_enable_x64", True)
from solid_harmonics import make_recursion

def diag_const(m):
    if m == 0: return 1.0
    num = np.prod([2*j-1 for j in range(1, m+1)], dtype=float)
    den = np.prod([2*j   for j in range(1, m+1)], dtype=float)
    return float(np.sqrt(2*num/den))

def make_periodic(k, a_max, L):
    """m = k*a only. Closed-form diagonal seeds + vertical chains."""
    ms = [k*a for a in range(a_max+1) if k*a <= L]
    table = [(l, s*m) for m in ms for l in range(m, L+1)
             for s in ([1] if m == 0 else [1, -1])]
    def f(x, y, z):
        r2 = x*x + y*y + z*z
        R = {}
        # diagonal seeds: powers of w^k
        wr, wi = x, y
        Wr, Wi = 1.0 + 0.0*x, 0.0*x
        for _ in range(k):                       # W = w^k
            Wr, Wi = Wr*wr - Wi*wi, Wr*wi + Wi*wr
        Pr, Pi = 1.0 + 0.0*x, 0.0*x
        for a, m in enumerate(ms):
            if a > 0:
                Pr, Pi = Pr*Wr - Pi*Wi, Pr*Wi + Pi*Wr
            c = diag_const(m)
            R[(m, m)] = c*Pr
            if m: R[(m, -m)] = c*Pi
        # vertical chains, independent per m
        for m in ms:
            for mm in ([0] if m == 0 else [m, -m]):
                for l in range(m+1, L+1):
                    a_ = (2*l-1)/np.sqrt(l*l - m*m)
                    t = a_*z*R[(l-1, mm)]
                    if l-1 > m:
                        b = np.sqrt((l-1)**2 - m*m)/np.sqrt(l*l - m*m)
                        t = t - b*r2*R[(l-2, mm)]
                    R[(l, mm)] = t
        return jnp.stack([R[t] for t in table])
    return jax.jit(f), table

k, a_max, L, N = 3, 6, 24, 50_000
fp, table = make_periodic(k, a_max, L)
full_table = [(l, m) for l in range(L+1) for m in range(-l, l+1)]
ff = make_recursion(full_table)

rng = np.random.default_rng(0)
_v = rng.normal(size=(3, N)); _v /= np.linalg.norm(_v, axis=0); x, y, z = (jnp.asarray(t) for t in _v)
A = fp(x, y, z); B = ff(x, y, z)
idx = {t: i for i, t in enumerate(full_table)}
Bsel = jnp.stack([B[idx[t]] for t in table])
print(f"k={k} a_max={a_max} L={L}: {len(table)} modes vs {len(full_table)} full")
print(f"max rel err vs verified full recursion: {float((jnp.abs(A-Bsel)/(jnp.abs(Bsel)+1e-12)).max()):.2e}\n")

for name, f, nm in [("full     ", ff, len(full_table)), ("periodic ", fp, len(table))]:
    for _ in range(3): f(x, y, z).block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10): f(x, y, z).block_until_ready()
    dt = (time.perf_counter()-t0)/10*1e3
    print(f"{name} {dt:7.1f} ms   ({nm} modes)")
