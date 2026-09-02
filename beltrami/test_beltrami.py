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
    find_beltrami_lam_axisym,
    generate_beltrami_callable,
    generate_beltrami_weighted_callable,
)
from beltrami.beltrami import _forcing, _jax_jm_djm, _jax_radial_rho, _precompute, _tail_exponent


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


# ---------------------------------------------------------------------------
# Axisymmetric (m = 0) Sec. II modes
# ---------------------------------------------------------------------------


def test_jax_jm_djm_m0_sign_and_axis():
    """J_0' = -J_1 (not +J_1 from js[-1] wraparound) and finite at z = 0."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import bessel_jn
    from scipy.special import jv, jvp

    z = jnp.array([0.0, 1e-6, 1e-4, 0.5, 1.3, 3.8317])
    jm, djm = _jax_jm_djm(jnp, bessel_jn, 0, z)
    jm_np = np.asarray(jm)
    djm_np = np.asarray(djm)
    assert np.all(np.isfinite(jm_np))
    assert np.all(np.isfinite(djm_np))
    np.testing.assert_allclose(jm_np, jv(0, np.asarray(z)), atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(
        djm_np, jvp(0, np.asarray(z), 1), atol=1e-10, rtol=1e-10
    )
    # Explicit sign check vs the wraparound bug (+J_1).
    np.testing.assert_allclose(djm_np, -jv(1, np.asarray(z)), atol=1e-10, rtol=1e-10)


def test_anharmonic_defaults_to_one():
    """Existing m >= 1 spectra keep anharmonic = 1.0 (bit-compatible)."""
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
    assert spec.anharmonic == 1.0
    assert spec.n_modes >= 1


def test_find_beltrami_lam_axisym_lam_values():
    from scipy.special import jn_zeros

    radius = 1.0
    spec = find_beltrami_lam_axisym(
        min_lam=4.0, max_lam=12.0, radius=radius, max_order=20
    )
    assert spec.m == 0
    assert spec.anharmonic == 0.0
    assert spec.n_modes >= 1
    assert np.all(spec.residual == 0.0)
    assert np.all(spec.status == int(IntervalStatus.ROOT))
    assert np.all(np.diff(spec.lam) >= 0.0)
    assert np.all((spec.lam >= 4.0) & (spec.lam <= 12.0))

    # Every accepted λ must match sqrt(j_{1,p}^2 + q^2 π^2) for some p, q.
    j1 = jn_zeros(1, 20)
    expected = []
    for j in j1:
        k = j / radius
        for q in range(1, 2 * 20 + 1):
            lam = np.sqrt(k * k + (q * np.pi) ** 2)
            if 4.0 <= lam <= 12.0:
                expected.append(lam)
    expected = np.sort(np.asarray(expected))
    np.testing.assert_allclose(spec.lam, expected, rtol=1e-12, atol=1e-12)

    # One-hot encoding: exactly one nonzero among an/bn per mode.
    for i in range(spec.n_modes):
        n_an = int(np.count_nonzero(spec.an[i]))
        n_bn = int(np.count_nonzero(spec.bn[i]))
        assert n_an + n_bn == 1
        assert abs(float(spec.an[i].sum() + spec.bn[i].sum()) - 1.0) < 1e-15


def test_find_beltrami_lam_axisym_filtering():
    wide = find_beltrami_lam_axisym(1.0, 20.0, radius=1.0, max_order=10)
    narrow = find_beltrami_lam_axisym(5.0, 8.0, radius=1.0, max_order=10)
    assert narrow.n_modes < wide.n_modes
    assert np.all((narrow.lam >= 5.0) & (narrow.lam <= 8.0))
    # Narrow window is a contiguous subset of the wide spectrum.
    for lam in narrow.lam:
        assert np.min(np.abs(wide.lam - lam)) < 1e-12


def test_axisym_field_curl_and_boundary():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    spec = find_beltrami_lam_axisym(
        min_lam=4.0, max_lam=8.0, radius=1.0, max_order=20
    )
    assert spec.n_modes >= 2
    fns = generate_beltrami_callable(spec)

    # Curl identity for the first two modes (covers both odd and even q).
    for i in range(min(2, spec.n_modes)):
        fn = fns[i]
        lam = float(spec.lam[i])

        def b_cart(xyz, _fn=fn):
            x, y, zz = xyz[0], xyz[1], xyz[2]
            return _fn(jnp.sqrt(x * x + y * y), jnp.atan2(y, x), zz)

        curl_fn = jax.jacfwd(b_cart)
        rels = []
        for pt in ([0.3, 0.2, 0.1], [0.5, -0.4, -0.25], [0.15, 0.6, 0.33]):
            xyz = jnp.array(pt, dtype=jnp.float64)
            bval = np.asarray(b_cart(xyz))
            jac = np.asarray(curl_fn(xyz))
            curl = np.array(
                [jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]]
            )
            rel = float(
                np.linalg.norm(curl - lam * bval)
                / max(np.linalg.norm(lam * bval), 1e-30)
            )
            rels.append(rel)
        assert max(rels) < 1e-8, f"mode {i} lam={lam}: {max(rels)}"

    # Boundary conditions via finite differences on χ reconstructed from B.
    # B_R(a) = 0 and B_Z(±1/2) = 0 (Sec. II).
    fn0 = fns[0]
    radius = float(spec.radius)

    def b_cyl(r, phi, z):
        bxyz = np.asarray(fn0(r, phi, z))
        # Invert Cartesian → cylindrical: B_R = Bx cos + By sin, B_φ = -Bx sin + By cos.
        cos_p, sin_p = np.cos(phi), np.sin(phi)
        br = bxyz[..., 0] * cos_p + bxyz[..., 1] * sin_p
        bz = bxyz[..., 2]
        return br, bz

    # B_R at R = a.
    for z in (-0.3, -0.1, 0.0, 0.2, 0.4):
        br, _ = b_cyl(radius, 0.0, z)
        assert abs(complex(br)) < 1e-8, f"B_R(a) at z={z}: {br}"

    # B_Z at Z = ±1/2.
    for r in (0.1, 0.4, 0.7, 0.95):
        for z_wall in (-0.5, 0.5):
            _, bz = b_cyl(r, 0.0, z_wall)
            assert abs(complex(bz)) < 1e-8, f"B_Z at r={r}, z={z_wall}: {bz}"


def test_axisym_field_finite_on_axis():
    spec = find_beltrami_lam_axisym(
        min_lam=4.0, max_lam=6.0, radius=1.0, max_order=20
    )
    assert spec.n_modes >= 1
    fn = generate_beltrami_callable(spec)[0]
    out = np.asarray(fn(0.0, 0.0, 0.0))
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    # Axisymmetric mode peaks on axis: |B_Z| should be nonzero.
    assert abs(complex(out[2])) > 1e-6


def test_find_beltrami_lam_axisym_validation():
    with pytest.raises(ValueError):
        find_beltrami_lam_axisym(min_lam=-1.0, max_lam=5.0, radius=1.0)
    with pytest.raises(ValueError):
        find_beltrami_lam_axisym(min_lam=5.0, max_lam=4.0, radius=1.0)
    with pytest.raises(ValueError):
        find_beltrami_lam_axisym(min_lam=1.0, max_lam=5.0, radius=0.0)
    with pytest.raises(ValueError):
        find_beltrami_lam_axisym(min_lam=1.0, max_lam=5.0, radius=1.0, max_order=1)


def test_jax_radial_rho_1d_unchanged_shape():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import bessel_jn, i0e, i1e

    r = jnp.linspace(0.1, 0.9, 5)
    kr2 = jnp.asarray([1.0, -2.0, 0.0, 4.0], dtype=jnp.float64)
    rho, drho = _jax_radial_rho(jnp, bessel_jn, i0e, i1e, 1, 1.0, kr2, r)
    assert rho.shape == (5, 4)
    assert drho.shape == (5, 4)
    assert np.all(np.isfinite(np.asarray(rho)))
    assert np.all(np.isfinite(np.asarray(drho)))


def test_jax_radial_rho_2d_kr2():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.scipy.special import bessel_jn, i0e, i1e

    r = jnp.linspace(0.1, 0.9, 5)
    kr2_1d = jnp.asarray([1.0, -2.0, 0.0], dtype=jnp.float64)
    kr2_2d = jnp.stack([kr2_1d, 2.0 * kr2_1d], axis=0)  # (2, 3)
    rho1, _ = _jax_radial_rho(jnp, bessel_jn, i0e, i1e, 1, 1.0, kr2_1d, r)
    rho2, _ = _jax_radial_rho(jnp, bessel_jn, i0e, i1e, 1, 1.0, kr2_2d, r)
    assert rho2.shape == (5, 2, 3)
    np.testing.assert_allclose(
        np.asarray(rho2[:, 0, :]), np.asarray(rho1), rtol=1e-12, atol=1e-12
    )


def test_weighted_callable_matches_per_mode_sum():
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
    fns = generate_beltrami_callable(spec)
    weighted = generate_beltrami_weighted_callable(spec)
    rng = np.random.default_rng(7)
    w = rng.normal(size=spec.n_modes)
    r = rng.uniform(0.1, 0.9, size=8)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=8)
    z = rng.uniform(-0.4, 0.4, size=8)
    expected = sum(float(w[i]) * np.asarray(fns[i](r, phi, z)) for i in range(spec.n_modes))
    got = np.asarray(weighted(r, phi, z, w))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_weighted_callable_real_matches_real_of_complex():
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
    w = np.ones(spec.n_modes)
    cpx = generate_beltrami_weighted_callable(spec, real=False)
    real = generate_beltrami_weighted_callable(spec, real=True)
    r, phi, z = 0.4, 0.2, 0.1
    np.testing.assert_allclose(
        np.asarray(real(r, phi, z, w)),
        np.real(np.asarray(cpx(r, phi, z, w))),
        rtol=1e-12,
        atol=1e-12,
    )
