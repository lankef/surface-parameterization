"""Self-check and benchmark for harmonic.py.

Run from the repo root::

    python -m tests.bench_harmonic
    # or
    python tests/bench_harmonic.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow importing harmonic.py from the repo root when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import sph_harm_y

from harmonic import HarmonicGEMM, HarmonicRecursion
from surface_transport.surface_transport import _make_harmonic


def scipy_ref(x, y, z, l, m_signed, irregular=False):
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.where(
        r > 0, np.arccos(np.clip(z / np.maximum(r, 1e-300), -1, 1)), 0.0
    )
    phi = np.arctan2(y, x)
    m = abs(m_signed)
    Y = sph_harm_y(l, m, theta, phi)
    radial = r ** (-(l + 1)) if irregular else r ** l
    pref = np.sqrt(4 * np.pi / (2 * l + 1)) * radial
    if m == 0:
        return pref * Y.real
    f = pref * np.sqrt(2.0) * ((-1) ** m)
    return f * (Y.real if m_signed > 0 else Y.imag)


def main():
    jax.config.update("jax_enable_x64", True)

    nfp, a_max, l_max = 3, 6, 24
    rng = np.random.default_rng(0)
    n = 200
    gamma_np = rng.normal(size=(n, 3))
    # Keep points away from the origin for irregular checks.
    gamma_np /= np.linalg.norm(gamma_np, axis=1, keepdims=True)
    gamma_np *= rng.uniform(0.5, 2.0, size=(n, 1))
    gamma = jnp.asarray(gamma_np)
    x_np, y_np, z_np = gamma_np[:, 0], gamma_np[:, 1], gamma_np[:, 2]
    r_np = np.linalg.norm(gamma_np, axis=1)

    print("=" * 60)
    print("0. mode-set sanity (zonal a=0 always present)")
    print("=" * 60)
    for stellsym in (True, False):
        h = HarmonicRecursion(nfp, a_max, l_max, stellsym=stellsym)
        assert (0, 0) not in h.keys
        zonal = {(l, 0) for l in range(1, l_max + 1)}
        assert zonal <= set(h.keys), f"missing zonal modes: {zonal - set(h.keys)}"
        assert h.irregular is True
        print(f"  stellsym={stellsym}: {h.num_modes} modes, "
              f"zonal l=1..{l_max} OK, irregular={h.irregular}")

    print()
    print("=" * 60)
    print("1. scipy cross-validation (regular)")
    print("=" * 60)
    for stellsym in (True, False):
        hr = HarmonicRecursion(
            nfp, a_max, l_max, stellsym=stellsym, irregular=False
        )
        hg = HarmonicGEMM(nfp, a_max, l_max, stellsym=stellsym)
        for name, h in (("HarmonicRecursion", hr), ("HarmonicGEMM", hg)):
            out = h.run_all(gamma)
            worst = 0.0
            for (l, a), got in out.items():
                exp = scipy_ref(x_np, y_np, z_np, l, nfp * a, irregular=False)
                rel = np.abs(np.asarray(got) - exp) / np.maximum(np.abs(exp), 1e-12)
                worst = max(worst, float(rel.max()))
            print(f"  {name:20s} stellsym={stellsym}: "
                  f"{h.num_modes} modes, worst rel err {worst:.3e}")

    print()
    print("=" * 60)
    print("1b. scipy cross-validation (irregular recursion)")
    print("=" * 60)
    for stellsym in (True, False):
        h = HarmonicRecursion(
            nfp, a_max, l_max, stellsym=stellsym, irregular=True
        )
        out = h.run_all(gamma)
        worst = 0.0
        for (l, a), got in out.items():
            exp = scipy_ref(x_np, y_np, z_np, l, nfp * a, irregular=True)
            rel = np.abs(np.asarray(got) - exp) / np.maximum(np.abs(exp), 1e-12)
            worst = max(worst, float(rel.max()))
        print(f"  HarmonicRecursion    stellsym={stellsym}: "
              f"{h.num_modes} modes, worst rel err {worst:.3e}")

    print()
    print("=" * 60)
    print("1c. I ≈ R / r^{2l+1}")
    print("=" * 60)
    for stellsym in (True, False):
        hr = HarmonicRecursion(
            nfp, a_max, l_max, stellsym=stellsym, irregular=False
        )
        hi = HarmonicRecursion(
            nfp, a_max, l_max, stellsym=stellsym, irregular=True
        )
        Ar, Ai = hr.run_all(gamma), hi.run_all(gamma)
        worst = 0.0
        for (l, a) in hr.keys:
            exp = np.asarray(Ar[(l, a)]) / (r_np ** (2 * l + 1))
            got = np.asarray(Ai[(l, a)])
            rel = np.abs(got - exp) / np.maximum(np.abs(exp), 1e-12)
            worst = max(worst, float(rel.max()))
        print(f"  stellsym={stellsym}: worst rel err {worst:.3e}")

    print()
    print("=" * 60)
    print("2. recursion vs GEMM + flat weighted (regular)")
    print("=" * 60)
    for stellsym in (True, False):
        hr = HarmonicRecursion(
            nfp, a_max, l_max, stellsym=stellsym, irregular=False
        )
        hg = HarmonicGEMM(nfp, a_max, l_max, stellsym=stellsym)
        Ar = hr.run_all(gamma)
        weights = {key: float(rng.normal()) for key in hr.keys}
        flat = hr.ravel(weights)
        wr = hr.run_weighted_dofs(gamma, flat)
        wg = hg.run_weighted_dofs(gamma, flat)
        ref = sum(weights[key] * Ar[key] for key in hr.keys)
        ref_a = np.asarray(ref)
        scale = np.maximum(np.abs(ref_a), 1e-12)
        print(f"  stellsym={stellsym}: flat vs sum(w*run_all) "
              f"rec rel={float((np.abs(np.asarray(wr)-ref_a)/scale).max()):.3e} "
              f"gemm rel={float((np.abs(np.asarray(wg)-ref_a)/scale).max()):.3e}")

    print()
    print("=" * 60)
    print("3. jacfwd grad vs FD")
    print("=" * 60)
    hr = HarmonicRecursion(
        nfp, a_max, min(l_max, 12), stellsym=False, irregular=True
    )
    hg = HarmonicGEMM(nfp, a_max, min(l_max, 12), stellsym=False)
    w = jnp.asarray(rng.normal(size=(hr.num_modes,)))
    p = jnp.asarray(rng.normal(size=(3,)))
    p = p / jnp.linalg.norm(p)  # away from origin

    g_exp = hr.grad_run_weighted_dofs(p, w)
    g_jaf = jax.jacfwd(lambda q: hr.run_weighted_dofs(q, w))(p)
    print(f"  irregular recursion grad vs jacfwd: "
          f"{float(jnp.abs(g_exp - g_jaf).max()):.3e}")

    w_g = jnp.asarray(rng.normal(size=(hg.num_modes,)))
    g_gem = hg.grad_run_weighted_dofs(p, w_g)
    g_gem_j = jax.jacfwd(lambda q: hg.run_weighted_dofs(q, w_g))(p)
    print(f"  gemm grad vs jacfwd:                "
          f"{float(jnp.abs(g_gem - g_gem_j).max()):.3e}")

    eps = 1e-6
    fd = jnp.zeros(3)
    for i in range(3):
        dp = jnp.zeros(3).at[i].set(eps)
        fd = fd.at[i].set(
            (hr.run_weighted_dofs(p + dp, w) - hr.run_weighted_dofs(p - dp, w))
            / (2 * eps)
        )
    print(f"  irregular recursion grad vs FD:     "
          f"{float(jnp.abs(g_exp - fd).max()):.3e}")

    gb = hr.grad_run_weighted_dofs(gamma[:10], w)
    assert gb.shape == (10, 3)

    print()
    print("=" * 60)
    print("3b. gemm + irregular rejected")
    print("=" * 60)
    try:
        _make_harmonic('gemm', nfp, a_max, l_max, True, True)
        raise AssertionError('expected ValueError')
    except ValueError as e:
        print(f"  OK: {e}")

    print()
    print("=" * 60)
    print("4. timings (N=100_000 run_all; N=1 grad)")
    print("=" * 60)
    N = 100_000
    _v = rng.normal(size=(N, 3))
    _v /= np.linalg.norm(_v, axis=1, keepdims=True)
    gamma_b = jnp.asarray(_v)
    w1 = jnp.ones((HarmonicRecursion(nfp, a_max, l_max).num_modes,))

    for label, h in (
        ("HarmonicRecursion irr", HarmonicRecursion(nfp, a_max, l_max, stellsym=True)),
        ("HarmonicRecursion reg", HarmonicRecursion(
            nfp, a_max, l_max, stellsym=True, irregular=False
        )),
        ("HarmonicGEMM", HarmonicGEMM(nfp, a_max, l_max, stellsym=True)),
    ):
        h.run_all(gamma_b)
        t0 = time.perf_counter()
        for _ in range(5):
            jax.tree_util.tree_map(
                lambda a: a.block_until_ready(), h.run_all(gamma_b)
            )
        print(f"  {label:24s} run_all  {(time.perf_counter()-t0)/5*1e3:7.1f} ms")

        p0 = gamma_b[0]
        ww = jnp.ones((h.num_modes,))
        h.grad_run_weighted_dofs(p0, ww).block_until_ready()
        t0 = time.perf_counter()
        for _ in range(50):
            h.grad_run_weighted_dofs(p0, ww).block_until_ready()
        print(f"  {label:24s} grad_1pt {(time.perf_counter()-t0)/50*1e3:7.1f} ms")

    print("\nOK")


if __name__ == "__main__":
    main()
