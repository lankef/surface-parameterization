"""Smoke tests for surface_distance.point_surface_distance_facet."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import jax.numpy as jnp
import numpy as np

from quadcoil.surface import SurfaceRZFourierJAX
from surface_distance import (
    _point_triangles_dist2,
    _triangulate_grid,
    point_surface_distance_facet,
)


def _circular_torus(nfp=2, R0=1.0, a=0.2, nphi=8, ntheta=8):
    """Axisymmetric circular torus as SurfaceRZFourierJAX (mpol=1, ntor=0)."""
    # stellsym dofs: [rc00, rc10, zs10] for mpol=1, ntor=0
    dofs = jnp.asarray([R0, a, a])
    return SurfaceRZFourierJAX(
        nfp=nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=jnp.linspace(0.0, 1.0 / nfp, nphi, endpoint=False),
        quadpoints_theta=jnp.linspace(0.0, 1.0, ntheta, endpoint=False),
        dofs=dofs,
    )


def test_output_shape():
    surf = _circular_torus()
    gamma = jnp.zeros((4, 5, 3))
    d2 = point_surface_distance_facet(surf, gamma, n_phi=6, n_theta=7)
    assert d2.shape == (4, 5)


def test_vertex_distance_near_zero():
    surf = _circular_torus(nfp=2)
    n_phi, n_theta = 8, 9
    surface_full = surf.copy_and_set_quadpoints(
        jnp.linspace(0.0, 1.0, surf.nfp * n_phi, endpoint=True),
        jnp.linspace(0.0, 1.0, n_theta, endpoint=True),
    )
    verts = surface_full.gamma()
    # A few mesh vertices must land exactly on triangles → d² ≈ 0
    queries = verts.reshape(-1, 3)[::17][:5]
    d2 = point_surface_distance_facet(surf, queries, n_phi=n_phi, n_theta=n_theta)
    assert jnp.all(d2 < 1e-20), f"vertex d2 not near zero: {d2}"


def test_far_point_matches_manual_min():
    surf = _circular_torus(nfp=1)
    n_phi, n_theta = 6, 7
    surface_full = surf.copy_and_set_quadpoints(
        jnp.linspace(0.0, 1.0, surf.nfp * n_phi, endpoint=True),
        jnp.linspace(0.0, 1.0, n_theta, endpoint=True),
    )
    verts = surface_full.gamma()
    triangles = _triangulate_grid(verts)

    # Point far outside the torus
    p = jnp.asarray([[10.0, 0.0, 0.0]])
    d2 = point_surface_distance_facet(surf, p, n_phi=n_phi, n_theta=n_theta)
    d2_ref = jnp.min(_point_triangles_dist2(p, triangles), axis=-1)
    np.testing.assert_allclose(np.asarray(d2), np.asarray(d2_ref), rtol=0, atol=1e-12)

    # Sanity: far from unit-scale torus, d² should be large
    assert float(d2[0]) > 50.0


def test_point_on_triangle_interior():
    """Midpoint of a triangle face should have d² ≈ 0."""
    a = jnp.array([0.0, 0.0, 0.0])
    b = jnp.array([1.0, 0.0, 0.0])
    c = jnp.array([0.0, 1.0, 0.0])
    tri = jnp.stack([a, b, c])[None, ...]
    p = ((a + b + c) / 3.0)[None, ...]
    d2 = _point_triangles_dist2(p, tri)
    assert float(d2[0, 0]) < 1e-20


def test_point_above_triangle():
    a = jnp.array([0.0, 0.0, 0.0])
    b = jnp.array([2.0, 0.0, 0.0])
    c = jnp.array([0.0, 2.0, 0.0])
    tri = jnp.stack([a, b, c])[None, ...]
    p = jnp.array([[0.5, 0.5, 3.0]])
    d2 = _point_triangles_dist2(p, tri)
    np.testing.assert_allclose(float(d2[0, 0]), 9.0, rtol=0, atol=1e-12)


if __name__ == '__main__':
    test_point_on_triangle_interior()
    test_point_above_triangle()
    test_output_shape()
    test_vertex_distance_near_zero()
    test_far_point_matches_manual_min()
    print('All surface_distance smoke tests passed.')
