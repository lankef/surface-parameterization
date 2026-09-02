"""Point-to-surface squared distance via a triangulated SurfaceJAX mesh."""

from functools import partial

import jax.numpy as jnp
from jax import jit


def _triangulate_grid(verts):
    """Split a structured (nphi, ntheta, 3) grid into two triangles per quad.

    Endpoints are assumed included, so wrap-around cells are ordinary neighbors
    (indices run over ``nphi-1`` and ``ntheta-1`` only).

    Returns
    -------
    triangles : jnp.ndarray, shape (T, 3, 3)
        ``T = 2 * (nphi-1) * (ntheta-1)``. Each triangle is three XYZ vertices.
    """
    nphi, ntheta, _ = verts.shape
    # Corners of each quad: (i,j), (i+1,j), (i,j+1), (i+1,j+1)
    v00 = verts[:-1, :-1]  # (nphi-1, ntheta-1, 3)
    v10 = verts[1:, :-1]
    v01 = verts[:-1, 1:]
    v11 = verts[1:, 1:]
    # T0 = (i,j), (i+1,j), (i,j+1)
    # T1 = (i+1,j), (i+1,j+1), (i,j+1)
    t0 = jnp.stack([v00, v10, v01], axis=-2)  # (..., 3, 3)
    t1 = jnp.stack([v10, v11, v01], axis=-2)
    triangles = jnp.stack([t0, t1], axis=0)  # (2, nphi-1, ntheta-1, 3, 3)
    return triangles.reshape(-1, 3, 3)


def _point_triangles_dist2(points, triangles):
    """Squared distance from each point to each triangle (Ericson RTCD 5.1.5).

    Parameters
    ----------
    points : jnp.ndarray, shape (N, 3)
    triangles : jnp.ndarray, shape (T, 3, 3)

    Returns
    -------
    d2 : jnp.ndarray, shape (N, T)
    """
    # Broadcast: points (N, 1, 3), triangle verts (1, T, 3)
    p = points[:, None, :]
    a = triangles[None, :, 0, :]
    b = triangles[None, :, 1, :]
    c = triangles[None, :, 2, :]

    ab = b - a
    ac = c - a
    ap = p - a

    d1 = jnp.sum(ab * ap, axis=-1)
    d2 = jnp.sum(ac * ap, axis=-1)

    # Region A (vertex)
    # Closest is A when d1 <= 0 and d2 <= 0
    dist2_a = jnp.sum(ap * ap, axis=-1)

    bp = p - b
    d3 = jnp.sum(ab * bp, axis=-1)
    d4 = jnp.sum(ac * bp, axis=-1)
    # Region B (vertex)
    dist2_b = jnp.sum(bp * bp, axis=-1)

    cp = p - c
    d5 = jnp.sum(ab * cp, axis=-1)
    d6 = jnp.sum(ac * cp, axis=-1)
    # Region C (vertex)
    dist2_c = jnp.sum(cp * cp, axis=-1)

    # Region AB (edge)
    vc = d1 * d4 - d3 * d2
    v_ab = d1 / jnp.where(d1 - d3 == 0, 1.0, d1 - d3)
    v_ab = jnp.clip(v_ab, 0.0, 1.0)
    closest_ab = a + v_ab[..., None] * ab
    dist2_ab = jnp.sum((p - closest_ab) ** 2, axis=-1)

    # Region AC (edge)
    vb = d5 * d2 - d1 * d6
    w_ac = d2 / jnp.where(d2 - d6 == 0, 1.0, d2 - d6)
    w_ac = jnp.clip(w_ac, 0.0, 1.0)
    closest_ac = a + w_ac[..., None] * ac
    dist2_ac = jnp.sum((p - closest_ac) ** 2, axis=-1)

    # Region BC (edge)
    va = d3 * d6 - d5 * d4
    bc = c - b
    w_bc = (d4 - d3) / jnp.where(
        (d4 - d3) + (d5 - d6) == 0, 1.0, (d4 - d3) + (d5 - d6)
    )
    w_bc = jnp.clip(w_bc, 0.0, 1.0)
    closest_bc = b + w_bc[..., None] * bc
    dist2_bc = jnp.sum((p - closest_bc) ** 2, axis=-1)

    # Region ABC (face)
    denom = va + vb + vc
    denom_safe = jnp.where(denom == 0, 1.0, denom)
    v_face = vb / denom_safe
    w_face = vc / denom_safe
    closest_face = a + v_face[..., None] * ab + w_face[..., None] * ac
    dist2_face = jnp.sum((p - closest_face) ** 2, axis=-1)

    # Region selection (Ericson barycentric tests)
    in_a = (d1 <= 0) & (d2 <= 0)
    in_b = (d3 >= 0) & (d4 <= d3)
    in_c = (d6 >= 0) & (d5 <= d6)
    in_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    in_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    in_bc = (va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0)
    # else face

    d2 = dist2_face
    d2 = jnp.where(in_bc, dist2_bc, d2)
    d2 = jnp.where(in_ac, dist2_ac, d2)
    d2 = jnp.where(in_ab, dist2_ab, d2)
    d2 = jnp.where(in_c, dist2_c, d2)
    d2 = jnp.where(in_b, dist2_b, d2)
    d2 = jnp.where(in_a, dist2_a, d2)
    return d2


@partial(jit, static_argnames=('n_phi', 'n_theta'))
def point_surface_distance_squared_facet(surface, gamma, n_phi, n_theta):
    """Squared distance from query points to a triangulated SurfaceJAX mesh.

    Resamples ``surface`` onto a closed full-torus grid (endpoints included),
    splits each quad into two triangles, and returns the minimum squared
    Euclidean distance from each point in ``gamma`` to any triangle.

    Parameters
    ----------
    surface : SurfaceJAX
        Any surface with ``copy_and_set_quadpoints`` and ``gamma()``.
    gamma : jnp.ndarray, shape (..., 3)
        Query points in Cartesian coordinates.
    n_phi, n_theta : int
        Grid resolution per field period (phi) and poloidally (theta).
        Static for JIT. Full phi grid has ``surface.nfp * n_phi`` points
        including the endpoint.

    Returns
    -------
    d2 : jnp.ndarray, shape gamma.shape[:-1]
        Unsigned squared distance to the nearest mesh triangle.
    """
    surface_full = surface.copy_and_set_quadpoints(
        jnp.linspace(0.0, 1.0, surface.nfp * n_phi, endpoint=True),
        jnp.linspace(0.0, 1.0, n_theta, endpoint=True),
    )
    verts = surface_full.gamma()
    triangles = _triangulate_grid(verts)

    leading = gamma.shape[:-1]
    points = gamma.reshape(-1, 3)
    d2_all = _point_triangles_dist2(points, triangles)
    d2 = jnp.min(d2_all, axis=-1)
    return d2.reshape(leading)

def surface_surface_distance(target_surface, free_surface, n_phi, n_theta):
    gamma = free_surface.gamma()
    distances_squared = point_surface_distance_squared_facet(target_surface, gamma, n_phi, n_theta)
    return free_surface.integrate(distances_squared)