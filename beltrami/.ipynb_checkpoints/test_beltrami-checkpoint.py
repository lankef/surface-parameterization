"""Tests for Morse (2005) Beltrami eigenvalue finder."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from beltrami import (
    IntervalStatus,
    evaluate_kernel,
    extrapolate_lam,
    find_beltrami_lam,
    generate_beltrami_callable,
)
from beltrami.beltrami import _forcing, _precompute, _tail_exponent


def test_argument_validation():
    kwargs = dict(
        m=1,
        min_lam=5.0,
        max_lam=6.0,
        n_lam=4,
        max_order=8,
        max_iter=20,
        tol=1e-8,
    )
    with pytest.raises(ValueError):
        find_beltrami_lam(**{**kwargs, "m": 0})
    with pytest.raises(ValueError):
        find_beltrami_lam(**{**kwargs, "min_lam": -1.0})
    with pytest.raises(ValueError):
        find_beltrami_lam(**{**kwargs, "max_lam": 4.0})
    with pytest.raises(ValueError):
        find_beltrami_lam(**{**kwargs, "n_lam": 0})
    with pytest.raises(ValueError):
        find_beltrami_lam(**{**kwargs, "max_order": 1})
    with pytest.raises(ValueError):
        find_beltrami_lam(**{**kwargs, "tol": 0.0})
    with pytest.raises(ValueError):
        find_beltrami_lam(**{**kwargs, "radius": 0.0})


def test_forcing_limits_at_p_pi():
    pre = _precompute(6)
    n = pre["n"]
    # λ = (2n0-1)π → u_cos[n0] → -1; λ = 2 n0 π → u_sin[n0] → -1
    for n0 in (1, 2, 3):
        u_cos, _ = _forcing(np.array([(2 * n0 - 1) * np.pi]), n)
        assert u_cos[0, n0 - 1] == pytest.approx(-1.0, abs=1e-12)
        _, u_sin = _forcing(np.array([2 * n0 * np.pi]), n)
        assert u_sin[0, n0 - 1] == pytest.approx(-1.0, abs=1e-12)


def test_eps_finite_across_j_to_i_branch():
    # For a=1, k_odd[1]^2 = λ^2 - π^2 changes sign near λ = π.
    lams = np.linspace(2.5, 4.0, 31)
    out = evaluate_kernel(1, lams, max_order=20, radius=1.0)
    assert np.all(np.isfinite(out["eps"]))
    assert np.any(out["kr2_odd"][:, 0] > 0.0)
    assert np.any(out["kr2_odd"][:, 0] < 0.0)


def test_sc_matches_direct_projection():
    pre = _precompute(6)
    sc = pre["sc"]
    z = np.linspace(-0.5, 0.5, 20001)
    dz = z[1] - z[0]

    def proj_sin(func, k):
        return 2.0 * np.sum(func * np.sin(2 * k * np.pi * z)) * dz

    n = 1
    dcos = -(2 * n - 1) * np.pi * np.sin((2 * n - 1) * np.pi * z)
    numeric = np.array([proj_sin(dcos, k) for k in range(1, 7)])
    np.testing.assert_allclose(numeric, -sc[0], atol=2e-3)


def test_unit_aspect_lowest_eigenvalue():
    spec = find_beltrami_lam(
        m=1,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=6,
        max_order=50,
        max_iter=40,
        tol=1e-10,
        radius=1.0,
    )
    assert spec.n_modes >= 1
    lam_over_pi = spec.lam[0] / np.pi
    assert lam_over_pi == pytest.approx(1.73426, abs=2e-5)
    assert abs(spec.residual[0]) < 1e-9
    assert spec.an.shape == (spec.n_modes, 50)
    assert spec.bn.shape == (spec.n_modes, 50)
    assert np.all(np.isfinite(spec.an))
    assert np.all(np.isfinite(spec.bn))
    kernel = evaluate_kernel(1, spec.lam[0], 50, radius=1.0)
    assert kernel["eps"] == pytest.approx(spec.residual[0], abs=1e-12)


def test_thin_disk_asymptote():
    spec = find_beltrami_lam(
        m=1,
        min_lam=3.180,
        max_lam=3.185,
        n_lam=20,
        max_order=40,
        max_iter=40,
        tol=1e-9,
        radius=10.0,
    )
    assert spec.n_modes >= 1
    # Morse thin-disk estimate √(π² + (5.13562/a)²) = 3.18265
    assert spec.lam[0] == pytest.approx(3.18265, abs=5e-3)
    assert spec.lam[0] == pytest.approx(3.18296, abs=5e-4)


def test_decay_laws_on_and_off_root():
    root = 1.7342624 * np.pi
    on = evaluate_kernel(1, root, 80, radius=1.0)
    off = evaluate_kernel(1, 4.30, 80, radius=1.0)
    n = np.arange(1, 81)
    exp_a_on = _tail_exponent(on["an"], n)
    exp_b_on = _tail_exponent(on["bn"], n)
    exp_a_off = _tail_exponent(off["an"], n)
    exp_b_off = _tail_exponent(off["bn"], n)
    assert exp_a_on == pytest.approx(-4.0, abs=1.2)
    assert exp_b_on == pytest.approx(-4.0, abs=1.2)
    assert exp_a_off == pytest.approx(-2.0, abs=0.8)
    assert exp_b_off < -3.0


def test_missing_root_intervals_are_reported():
    spec = find_beltrami_lam(
        m=1,
        min_lam=4.0,
        max_lam=4.4,
        n_lam=8,
        max_order=20,
        max_iter=20,
        tol=1e-8,
        radius=1.0,
    )
    assert spec.n_modes == 0
    assert spec.status.shape == (8,)
    assert np.all(
        (spec.status == int(IntervalStatus.NO_SIGN_CHANGE))
        | (spec.status == int(IntervalStatus.NONFINITE))
        | (spec.status == int(IntervalStatus.POLE_CROSSING))
        | (spec.status == int(IntervalStatus.RESIDUAL_TOO_LARGE))
        | (spec.status == int(IntervalStatus.MAX_ITER))
    )


def test_spurious_pi_root_is_filtered():
    # For a=0.05, ε crosses zero smoothly at λ = 20π.
    target = 20.0 * np.pi
    spec = find_beltrami_lam(
        m=1,
        min_lam=target - 0.04,
        max_lam=target + 0.04,
        n_lam=4,
        max_order=40,
        max_iter=40,
        tol=1e-8,
        radius=0.05,
        pi_guard=1e-3,
    )
    assert spec.n_modes == 0
    assert int(IntervalStatus.SPURIOUS_PI) in set(spec.status.tolist())


def test_dedup_collapses_neighbor_brackets():
    spec = find_beltrami_lam(
        m=1,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=40,
        max_order=25,
        max_iter=30,
        tol=1e-8,
        radius=1.0,
    )
    assert spec.n_modes >= 1
    if spec.n_modes > 1:
        gaps = np.diff(spec.lam)
        assert np.all(gaps > 1e-6)
    n_root = int(np.sum(spec.status == int(IntervalStatus.ROOT)))
    n_dup = int(np.sum(spec.status == int(IntervalStatus.DUPLICATE)))
    assert n_root == spec.n_modes
    assert n_root + n_dup >= 1


def test_convergence_and_richardson():
    orders = np.array([25, 50, 100])
    lams = []
    for n in orders:
        spec = find_beltrami_lam(
            m=1,
            min_lam=5.30,
            max_lam=5.60,
            n_lam=4,
            max_order=int(n),
            max_iter=40,
            tol=1e-12,
            radius=1.0,
        )
        assert spec.n_modes == 1
        lams.append(spec.lam[0])
    lams = np.asarray(lams)
    # Error shrinks by about 4× when N doubles (N^{-2}).
    err_ratio = abs(lams[1] - lams[2]) / abs(lams[0] - lams[1])
    assert err_ratio == pytest.approx(0.25, abs=0.12)
    extra = extrapolate_lam(orders, lams)
    assert abs(extra - lams[-1]) < abs(lams[-2] - lams[-1])
    assert extra / np.pi == pytest.approx(1.7342624, abs=5e-6)


def test_spectrum_fields():
    spec = find_beltrami_lam(
        m=1,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=6,
        max_order=20,
        max_iter=30,
        tol=1e-8,
        radius=1.0,
    )
    assert spec.n_modes >= 1
    assert spec.m == 1
    assert spec.radius == 1.0
    assert spec.max_order == 20
    assert spec.lam.shape == (spec.n_modes,)
    assert spec.an.shape == (spec.n_modes, 20)
    assert spec.kz_even.shape == (20,)
    assert spec.status.shape == (6,)
    assert int(np.sum(spec.status == int(IntervalStatus.ROOT))) == spec.n_modes


def test_bessel_denom_is_positive_at_root():
    spec = find_beltrami_lam(
        m=1,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=4,
        max_order=30,
        max_iter=30,
        tol=1e-9,
        radius=1.0,
    )
    assert spec.n_modes == 1
    assert spec.min_bessel_denom[0] > 0.0
    kr2 = spec.kr2_even[0]
    real = kr2 > 0.0
    if np.any(real):
        k = np.sqrt(kr2[real])
        den = np.abs(jv(1, k * 1.0))
        assert spec.min_bessel_denom[0] <= den.min() + 1e-15


def _cfmt(value: complex) -> str:
    return f"{value.real:+.4e}{value.imag:+.4e}j"


def test_generated_callables_satisfy_beltrami_equation():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = find_beltrami_lam(
        m=1,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=6,
        max_order=40,
        max_iter=40,
        tol=1e-10,
        radius=1.0,
    )
    fns = generate_beltrami_callable(spec)
    assert len(fns) == spec.n_modes
    assert len(fns) != spec.status.shape[0]
    assert len(fns) >= 1

    fn = fns[0]
    lam = float(spec.lam[0])
    radius = float(spec.radius)
    rng = np.random.default_rng(20260830)
    r = rng.uniform(0.05 * radius, 0.90 * radius, size=20)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=20)
    z = rng.uniform(-0.40, 0.40, size=20)

    batched = np.asarray(fn(r, phi, z))
    assert batched.shape == (20, 3)
    assert np.iscomplexobj(batched)

    def b_cart(xyz):
        x, y, zz = xyz[0], xyz[1], xyz[2]
        rr = jnp.sqrt(x * x + y * y)
        pp = jnp.atan2(y, x)
        return fn(rr, pp, zz)

    curl_fn = jax.jacfwd(b_cart)

    rels = []
    print("\nBeltrami check: curl B vs λB  (m=1, a=1, lowest mode)")
    print(
        f"{'i':>2} {'R':>8} {'phi':>8} {'Z':>8} "
        f"{'curl_x':>24} {'rhs_x':>24} {'err_x':>24} {'rel':>10}"
    )
    print(
        f"{'':>2} {'x':>8} {'y':>8} {'z':>8} "
        f"{'curl_y':>24} {'rhs_y':>24} {'err_y':>24} {'':>10}"
    )
    print(
        f"{'':>2} {'':>8} {'':>8} {'':>8} "
        f"{'curl_z':>24} {'rhs_z':>24} {'err_z':>24} {'':>10}"
    )
    for i in range(20):
        x = r[i] * np.cos(phi[i])
        y = r[i] * np.sin(phi[i])
        xyz = jnp.array([x, y, z[i]], dtype=jnp.float64)
        bval = np.asarray(b_cart(xyz))
        jac = np.asarray(curl_fn(xyz))
        curl = np.array(
            [jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]]
        )
        rhs = lam * bval
        err = curl - rhs
        rel = float(np.linalg.norm(err) / max(np.linalg.norm(rhs), 1e-30))
        rels.append(rel)
        print(
            f"{i:2d} {r[i]:8.4f} {phi[i]:8.4f} {z[i]:8.4f} "
            f"{_cfmt(curl[0]):>24} {_cfmt(rhs[0]):>24} {_cfmt(err[0]):>24} {rel:10.3e}"
        )
        print(
            f"{'':2} {x:8.4f} {y:8.4f} {z[i]:8.4f} "
            f"{_cfmt(curl[1]):>24} {_cfmt(rhs[1]):>24} {_cfmt(err[1]):>24}"
        )
        print(
            f"{'':2} {'':8} {'':8} {'':8} "
            f"{_cfmt(curl[2]):>24} {_cfmt(rhs[2]):>24} {_cfmt(err[2]):>24}"
        )

    max_rel = max(rels)
    print(f"\nmax relative |curl B - λB| / |λB| = {max_rel:.3e}")
    assert max_rel < 1e-4
