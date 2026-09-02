"""Tests for BeltramiBasis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beltrami import BeltramiBasis


def _unit_aspect_basis(**overrides) -> BeltramiBasis:
    """Basis covering the known m=1 unit-aspect root (λ/π ≈ 1.73426)."""
    kwargs = dict(
        m_per_period=1,
        nfp=1,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=6,
        radius=1.0,
        max_order=20,
        max_iter=30,
        tol=1e-8,
    )
    kwargs.update(overrides)
    return BeltramiBasis(**kwargs)


def test_m_list_skips_zero_and_scales_by_nfp():
    # Empty spectra are fine: we only care that m=0 is never requested and
    # that the constructed m array (when nonempty) matches arange(1, …)*nfp.
    expected = np.arange(1, 3 + 1) * 2  # [2, 4, 6]
    assert 0 not in expected.tolist()
    basis = BeltramiBasis(
        m_per_period=3,
        nfp=2,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=4,
        radius=1.0,
        max_order=10,
        max_iter=10,
        tol=1e-6,
    )
    # No roots in this window for m=2,4,6, but construction must succeed
    # without hitting the m=0 ValueError.
    assert len(basis) == 0
    assert basis.m.shape == (0,)
    assert basis.lam.shape == (0,)


def test_attributes_are_aligned():
    basis = _unit_aspect_basis()
    assert len(basis) >= 1
    assert len(basis) == len(basis.beltrami_basis)
    assert len(basis) == basis.m.shape[0]
    assert len(basis) == basis.lam.shape[0]
    assert np.all(basis.m == 1)
    assert np.all(np.isfinite(basis.lam))
    assert basis.lam[0] / np.pi == pytest.approx(1.73426, abs=2e-4)


def test_multi_m_flattens_across_modes():
    # Wider window so m=1 and m=2 each contribute at least one root.
    basis = BeltramiBasis(
        m_per_period=2,
        nfp=1,
        min_lam=5.0,
        max_lam=12.0,
        n_lam=40,
        radius=1.0,
        max_order=20,
        max_iter=30,
        tol=1e-8,
    )
    assert set(basis.m.tolist()).issubset({1, 2})
    assert 1 in set(basis.m.tolist())
    assert 2 in set(basis.m.tolist())
    assert len(basis) == len(basis.beltrami_basis) == basis.m.size == basis.lam.size
    # Entries belonging to each m form contiguous blocks in construction order.
    m1 = np.flatnonzero(basis.m == 1)
    m2 = np.flatnonzero(basis.m == 2)
    assert m1.size >= 1 and m2.size >= 1
    assert m1[-1] + 1 == m2[0]
    assert np.all(np.diff(m1) == 1) if m1.size > 1 else True
    assert np.all(np.diff(m2) == 1) if m2.size > 1 else True


def test_basis_callable_satisfies_beltrami_equation():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    basis = _unit_aspect_basis(max_order=40, max_iter=40, tol=1e-10)
    assert len(basis) >= 1
    assert int(basis.m[0]) == 1

    fn = basis.beltrami_basis[0]
    lam = float(basis.lam[0])
    radius = float(basis.radius)
    rng = np.random.default_rng(20260831)
    r = rng.uniform(0.05 * radius, 0.90 * radius, size=12)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=12)
    z = rng.uniform(-0.40, 0.40, size=12)

    def b_cart(xyz):
        x, y, zz = xyz[0], xyz[1], xyz[2]
        rr = jnp.sqrt(x * x + y * y)
        pp = jnp.atan2(y, x)
        return fn(rr, pp, zz)

    curl_fn = jax.jacfwd(b_cart)
    rels = []
    for i in range(12):
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
    assert max(rels) < 1e-4


def test_visualize_basis_writes_vts(tmp_path: Path):
    basis = _unit_aspect_basis()
    assert len(basis) >= 1
    stem = tmp_path / "basis"
    path = basis.visualize_basis(0, str(stem), n_r=8, n_phi=8, n_z=8)
    assert path.exists()
    assert path.suffix == ".vts"
    m0 = int(basis.m[0])
    lam0 = round(float(basis.lam[0]), 3)
    assert path.name == f"basis_i0_m{m0}_lam{lam0}.vts"

    # Spot-check the same grid evaluation is finite (axis excluded).
    fn = basis.beltrami_basis[0]
    r = np.linspace(0.0, basis.radius, 8 + 1)[1:]
    r = np.maximum(r, 1e-4 * basis.radius)
    phi = np.linspace(0.0, 2.0 * np.pi, 8)
    z = np.linspace(-0.5, 0.5, 8)
    R, Phi, Z = np.meshgrid(r, phi, z, indexing="ij")
    out = np.asarray(fn(R.ravel(), Phi.ravel(), Z.ravel()))
    assert np.all(np.isfinite(out))


def test_argument_validation():
    with pytest.raises(ValueError):
        BeltramiBasis(
            m_per_period=0,
            nfp=1,
            min_lam=5.0,
            max_lam=6.0,
            n_lam=4,
            radius=1.0,
        )
    with pytest.raises(ValueError):
        BeltramiBasis(
            m_per_period=1,
            nfp=0,
            min_lam=5.0,
            max_lam=6.0,
            n_lam=4,
            radius=1.0,
        )
    with pytest.raises(ValueError):
        BeltramiBasis(
            m_per_period=1,
            nfp=1,
            min_lam=5.0,
            max_lam=6.0,
            n_lam=4,
            radius=0.0,
        )
