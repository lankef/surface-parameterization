"""Tests for BeltramiBasis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from beltrami import BeltramiBasis, find_beltrami_lam_axisym


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


def test_m_list_includes_zero_and_scales_by_nfp():
    # m_list = arange(0, m_per_period+1)*nfp includes m=0. Window is chosen
    # so no roots fall in range for any of these m; construction must still
    # succeed (find_beltrami_lam_axisym for m=0, find_beltrami_lam otherwise).
    expected_m_values = np.arange(0, 3 + 1) * 2  # [0, 2, 4, 6]
    assert 0 in expected_m_values.tolist()
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
    # No roots in this window for m=0,2,4,6 (lowest m=0 root is ~4.95).
    assert len(basis) == 0
    assert basis.m.shape == (0,)
    assert basis.lam.shape == (0,)


def test_attributes_are_aligned():
    basis = _unit_aspect_basis()
    assert len(basis) >= 1
    assert len(basis) == len(basis.beltrami_basis)
    assert len(basis) == basis.m.shape[0]
    assert len(basis) == basis.lam.shape[0]
    # Window [5.30, 5.60] catches the m=1 unit-aspect root but no m=0 modes.
    assert np.all(basis.m == 1)
    assert np.all(np.isfinite(basis.lam))
    assert basis.lam[0] / np.pi == pytest.approx(1.73426, abs=2e-4)


def test_multi_m_flattens_across_modes():
    # Wider window so m=0, m=1 and m=2 each contribute at least one root.
    basis = BeltramiBasis(
        m_per_period=2,
        nfp=1,
        min_lam=4.5,
        max_lam=12.0,
        n_lam=40,
        radius=1.0,
        max_order=20,
        max_iter=30,
        tol=1e-8,
    )
    assert set(basis.m.tolist()).issubset({0, 1, 2})
    assert 0 in set(basis.m.tolist())
    assert 1 in set(basis.m.tolist())
    assert 2 in set(basis.m.tolist())
    assert len(basis) == len(basis.beltrami_basis) == basis.m.size == basis.lam.size
    # Entries belonging to each m form contiguous blocks in construction order.
    m0 = np.flatnonzero(basis.m == 0)
    m1 = np.flatnonzero(basis.m == 1)
    m2 = np.flatnonzero(basis.m == 2)
    assert m0.size >= 1 and m1.size >= 1 and m2.size >= 1
    assert m0[-1] + 1 == m1[0]
    assert m1[-1] + 1 == m2[0]
    assert np.all(np.diff(m0) == 1) if m0.size > 1 else True
    assert np.all(np.diff(m1) == 1) if m1.size > 1 else True
    assert np.all(np.diff(m2) == 1) if m2.size > 1 else True


def test_m0_visualize_includes_axis(tmp_path: Path):
    basis = BeltramiBasis(
        m_per_period=1,
        nfp=1,
        min_lam=4.5,
        max_lam=5.2,
        n_lam=4,
        radius=1.0,
        max_order=20,
        max_iter=10,
        tol=1e-6,
    )
    m0_idx = int(np.flatnonzero(basis.m == 0)[0])
    path = basis.visualize_basis(
        m0_idx, str(tmp_path / "m0"), n_r=6, n_phi=4, n_z=4
    )
    assert path.exists()

    # Spot-check: field is finite at r = 0 for the m=0 mode.
    fn = basis.beltrami_basis[m0_idx]
    out = np.asarray(fn(0.0, 0.0, 0.0))
    assert np.all(np.isfinite(out))
    assert abs(complex(out[2])) > 1e-6


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
    phi = np.linspace(0.0, 2.0 * np.pi / basis.nfp, 8)
    z = np.linspace(-0.5, 0.5, 8)
    R, Phi, Z = np.meshgrid(r, phi, z, indexing="ij")
    out = np.asarray(fn(R.ravel(), Phi.ravel(), Z.ravel()))
    assert np.all(np.isfinite(out))


def test_visualize_basis_exports_one_nfp_wedge(tmp_path: Path, monkeypatch):
    basis = BeltramiBasis(
        m_per_period=1,
        nfp=2,
        min_lam=8.0,
        max_lam=8.4,
        n_lam=8,
        radius=1.0,
        max_order=20,
        max_iter=30,
        tol=1e-8,
    )
    assert len(basis) >= 1

    captured = {}

    def fake_grid_to_vtk(stem, x, y, z, pointData):
        captured.update(x=x, y=y, z=z, pointData=pointData)
        path = Path(f"{stem}.vts")
        path.touch()
        return str(path)

    monkeypatch.setattr("pyevtk.hl.gridToVTK", fake_grid_to_vtk)
    basis.visualize_basis(
        0, str(tmp_path / "wedge"), n_r=4, n_phi=5, n_z=4
    )

    # The first and last phi rows are the two boundaries of one π-wide
    # wedge for nfp=2, not a full 2π cylinder.
    x = captured["x"]
    y = captured["y"]
    np.testing.assert_allclose(y[:, 0, :], 0.0, atol=1e-14)
    np.testing.assert_allclose(y[:, -1, :], 0.0, atol=1e-14)
    assert np.all(x[:, 0, :] > 0.0)
    assert np.all(x[:, -1, :] < 0.0)


def _rotation_matrix(phi_s: float) -> np.ndarray:
    c, s = np.cos(2.0 * phi_s), np.sin(2.0 * phi_s)
    return np.array([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]])


def test_stellsym_false_is_complex_and_keeps_even_q():
    full = BeltramiBasis(
        m_per_period=1,
        nfp=1,
        min_lam=4.5,
        max_lam=8.0,
        n_lam=20,
        radius=1.0,
        max_order=20,
        max_iter=30,
        tol=1e-8,
        stellsym=False,
    )
    assert full.stellsym is False
    assert len(full) >= 2
    spec0 = find_beltrami_lam_axisym(4.5, 8.0, radius=1.0, max_order=20)
    even_q = spec0.lam[np.any(spec0.an != 0.0, axis=1)]
    odd_q = spec0.lam[~np.any(spec0.an != 0.0, axis=1)]
    assert even_q.size >= 1 and odd_q.size >= 1
    m0 = full.lam[full.m == 0]
    np.testing.assert_allclose(np.sort(m0), np.sort(spec0.lam), atol=1e-12)
    sample = np.asarray(full.beltrami_basis[0](0.4, 0.2, 0.1))
    assert np.iscomplexobj(sample)
    assert sample.dtype == np.complex128


def test_stellsym_filters_even_q_and_returns_float64():
    kwargs = dict(
        m_per_period=1,
        nfp=1,
        min_lam=4.5,
        max_lam=8.0,
        n_lam=20,
        radius=1.0,
        max_order=20,
        max_iter=30,
        tol=1e-8,
    )
    full = BeltramiBasis(**kwargs, stellsym=False)
    sym = BeltramiBasis(**kwargs, stellsym=True)
    assert sym.stellsym is True

    spec0 = find_beltrami_lam_axisym(4.5, 8.0, radius=1.0, max_order=20)
    odd_q = spec0.lam[~np.any(spec0.an != 0.0, axis=1)]
    np.testing.assert_allclose(
        np.sort(sym.lam[sym.m == 0]), np.sort(odd_q), atol=1e-12
    )
    # m > 0 counts are unchanged; only the m=0 even-q modes are dropped.
    assert np.count_nonzero(sym.m == 1) == np.count_nonzero(full.m == 1)
    assert len(sym) == len(full) - (np.count_nonzero(full.m == 0) - odd_q.size)

    for fn in sym.beltrami_basis:
        sample = np.asarray(fn(0.4, 0.2, 0.1))
        assert sample.dtype == np.float64
        assert not np.iscomplexobj(sample)
        assert np.max(np.abs(sample)) > 0.0


def test_stellsym_fields_are_anti_equivariant_and_beltrami():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    nfp = 2
    basis = BeltramiBasis(
        m_per_period=1,
        nfp=nfp,
        min_lam=4.5,
        max_lam=9.0,
        n_lam=30,
        radius=1.0,
        max_order=20,
        max_iter=30,
        tol=1e-8,
        stellsym=True,
    )
    assert len(basis) >= 1
    assert 0 in set(basis.m.tolist())

    Q = _rotation_matrix(np.pi / nfp)
    rng = np.random.default_rng(20260831)
    r = rng.uniform(0.15, 0.85, size=6)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=6)
    z = rng.uniform(-0.35, 0.35, size=6)
    X = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)
    QX = X @ Q.T
    r2 = np.hypot(QX[:, 0], QX[:, 1])
    phi2 = np.arctan2(QX[:, 1], QX[:, 0])
    z2 = QX[:, 2]

    for i, fn in enumerate(basis.beltrami_basis):
        B = np.asarray(fn(r, phi, z))
        BQ = np.asarray(fn(r2, phi2, z2))
        assert B.dtype == np.float64
        scale = max(float(np.max(np.abs(B))), 1e-30)
        rel = float(np.max(np.abs(BQ + B @ Q.T)) / scale)
        assert rel < 1e-12, f"mode {i} m={int(basis.m[i])} lam={basis.lam[i]}: {rel}"

        def b_cart(xyz, _fn=fn):
            x, y, zz = xyz[0], xyz[1], xyz[2]
            return _fn(jnp.sqrt(x * x + y * y), jnp.atan2(y, x), zz)

        curl_fn = jax.jacfwd(b_cart)
        lam = float(basis.lam[i])
        xyz = jnp.array([X[0, 0], X[0, 1], X[0, 2]], dtype=jnp.float64)
        bval = np.asarray(b_cart(xyz))
        jac = np.asarray(curl_fn(xyz))
        curl = np.array(
            [jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]]
        )
        rel_curl = float(
            np.linalg.norm(curl - lam * bval)
            / max(np.linalg.norm(lam * bval), 1e-30)
        )
        assert rel_curl < 1e-8, f"curl mode {i}: {rel_curl}"


def test_stellsym_visualize_writes_real_vector(tmp_path: Path, monkeypatch):
    basis = BeltramiBasis(
        m_per_period=1,
        nfp=1,
        min_lam=4.5,
        max_lam=5.2,
        n_lam=4,
        radius=1.0,
        max_order=20,
        max_iter=10,
        tol=1e-6,
        stellsym=True,
    )
    assert len(basis) >= 1
    captured = {}

    def fake_grid_to_vtk(stem, x, y, z, pointData):
        captured.update(pointData=pointData)
        path = Path(f"{stem}.vts")
        path.touch()
        return str(path)

    monkeypatch.setattr("pyevtk.hl.gridToVTK", fake_grid_to_vtk)
    basis.visualize_basis(0, str(tmp_path / "sym"), n_r=4, n_phi=4, n_z=4)
    assert "B" in captured["pointData"]
    assert "B_magnitude" in captured["pointData"]
    assert "B_imag" not in captured["pointData"]
    assert "B_real" not in captured["pointData"]
    bx, by, bz = captured["pointData"]["B"]
    assert bx.dtype == np.float64
    assert np.all(np.isfinite(bx))


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


def test_eval_matches_per_mode_sum():
    basis = _unit_aspect_basis()
    assert len(basis) >= 1
    rng = np.random.default_rng(11)
    w = rng.normal(size=len(basis))
    r = rng.uniform(0.1, 0.9, size=6)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=6)
    z = rng.uniform(-0.4, 0.4, size=6)
    expected = sum(
        float(w[i]) * np.asarray(basis.beltrami_basis[i](r, phi, z))
        for i in range(len(basis))
    )
    got = np.asarray(basis.eval(r, phi, z, w))
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-10)


def test_eval_weights_length_mismatch_raises():
    basis = _unit_aspect_basis()
    with pytest.raises(ValueError, match="weights has length"):
        basis.eval(0.4, 0.2, 0.1, np.ones(len(basis) + 1))


def test_visualize_mixed_weights(tmp_path: Path, monkeypatch):
    basis = _unit_aspect_basis()
    assert len(basis) >= 1
    w = np.zeros(len(basis))
    w[0] = 1.5
    if len(basis) > 1:
        w[1] = -0.5

    captured = {}

    def fake_grid_to_vtk(stem, x, y, z, pointData):
        captured.update(stem=stem, pointData=pointData)
        path = Path(f"{stem}.vts")
        path.touch()
        return str(path)

    monkeypatch.setattr("pyevtk.hl.gridToVTK", fake_grid_to_vtk)
    path = basis.visualize(w, str(tmp_path / "mixed"), n_r=4, n_phi=4, n_z=4)
    assert path.exists()
    assert path.name == "mixed.vts"
    assert "B_real" in captured["pointData"]
    assert "B_imag" in captured["pointData"]


def test_visualize_basis_still_one_hot_filename(tmp_path: Path, monkeypatch):
    basis = _unit_aspect_basis()
    captured = {}

    def fake_grid_to_vtk(stem, x, y, z, pointData):
        captured["stem"] = stem
        path = Path(f"{stem}.vts")
        path.touch()
        return str(path)

    monkeypatch.setattr("pyevtk.hl.gridToVTK", fake_grid_to_vtk)
    path = basis.visualize_basis(0, str(tmp_path / "basis"), n_r=4, n_phi=4, n_z=4)
    m0 = int(basis.m[0])
    lam0 = round(float(basis.lam[0]), 3)
    assert path.name == f"basis_i0_m{m0}_lam{lam0}.vts"
    assert captured["stem"].endswith(f"basis_i0_m{m0}_lam{lam0}")
