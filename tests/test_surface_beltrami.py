"""Tests for SurfaceBeltramiJAX."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import tree_util

from surface_transport import SurfaceBeltramiJAX


def _tiny_surface(**overrides):
    """Small, fast Beltrami surface for unit tests."""
    kwargs = dict(
        nfp=1,
        stellsym=True,
        seed_major_radius=1.0,
        seed_minor_radius=0.2,
        quadpoints_phi=jnp.linspace(0.0, 1.0, 4, endpoint=False),
        quadpoints_theta=jnp.linspace(0.0, 1.0, 4, endpoint=False),
        step_num=2,
        m_per_period=1,
        min_lam=5.30,
        max_lam=5.60,
        n_lam=6,
        max_order=10,
        max_iter=20,
        tol=1e-6,
    )
    kwargs.update(overrides)
    return SurfaceBeltramiJAX(**kwargs)


def test_stellsym_false_raises():
    with pytest.raises(NotImplementedError, match="stellsym=True"):
        _tiny_surface(stellsym=False)


def test_dofs_length_validation():
    surf = _tiny_surface()
    n = len(surf.basis)
    assert surf.dofs.shape == (n,)
    with pytest.raises(ValueError, match="dofs has the wrong shape"):
        _tiny_surface(dofs=jnp.zeros((n + 1,)))


def test_pytree_roundtrip_hits_cache():
    surf = _tiny_surface()
    basis_id = id(surf.basis)
    leaves, treedef = tree_util.tree_flatten(surf)
    restored = tree_util.tree_unflatten(treedef, leaves)
    # tree_unflatten rebuilds via __init__; lru_cache must reuse the basis.
    assert id(restored.basis) == basis_id
    assert restored.box_r == surf.box_r
    assert restored.box_z == surf.box_z
    np.testing.assert_allclose(
        np.asarray(restored.dofs), np.asarray(surf.dofs)
    )


def test_gammadash_at_point_smoke():
    surf = _tiny_surface()
    # Nonzero dofs so the flow is nontrivial.
    dofs = jnp.zeros_like(surf.dofs)
    if dofs.size > 0:
        dofs = dofs.at[0].set(0.01)
    surf = _tiny_surface(dofs=dofs)
    phi = jnp.asarray(0.1)
    theta = jnp.asarray(0.2)
    g = surf.gammadash_at_point(phi, theta, 0, 0)
    assert g.shape == (3,)
    assert np.all(np.isfinite(np.asarray(g)))
    # First derivatives should also be finite.
    gp = surf.gammadash_at_point(phi, theta, 1, 0)
    gt = surf.gammadash_at_point(phi, theta, 0, 1)
    assert gp.shape == (3,) and gt.shape == (3,)
    assert np.all(np.isfinite(np.asarray(gp)))
    assert np.all(np.isfinite(np.asarray(gt)))


def test_default_box_from_seed():
    surf = _tiny_surface()
    assert surf.box_r == pytest.approx(1.0 + 2.0 * 0.2)
    assert surf.box_z == pytest.approx(2.0 * 0.2)
    assert surf.basis.radius == pytest.approx(surf.box_r / surf.box_z)
    assert surf.basis.stellsym is True
