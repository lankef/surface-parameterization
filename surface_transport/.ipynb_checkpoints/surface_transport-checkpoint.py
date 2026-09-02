from functools import lru_cache, partial

import jax.numpy as jnp
from jax import jacfwd, jit, lax, tree_util, vmap

from quadcoil.surface import SurfaceJAX, SurfaceRZFourierJAX

from harmonic import HarmonicGEMM, HarmonicRecursion

_BACKENDS = {
    'recursion': HarmonicRecursion,
    'gemm': HarmonicGEMM,
}


@lru_cache(maxsize=None)
def _make_harmonic(backend, nfp, a_max, l_max, stellsym, irregular):
    """Memoized Harmonic construction (static aux data only)."""
    Cls = _BACKENDS[backend]
    if Cls is HarmonicGEMM and irregular:
        raise ValueError("backend='gemm' does not support irregular=True")
    kwargs = dict(stellsym=bool(stellsym))
    if Cls is HarmonicRecursion:
        kwargs['irregular'] = bool(irregular)
    return Cls(int(nfp), int(a_max), int(l_max), **kwargs)


@tree_util.register_pytree_node_class
class SurfaceTransportJAX(SurfaceJAX):
    """Toroidal surface obtained by transporting a circular RZFourier seed.

    Subclasses implement :meth:`_gamma_map_single`. Geometry (normals, area,
    etc.) is inherited from :class:`SurfaceJAX` via :meth:`gammadash_at_point`.

    The circular seed is always stellarator-symmetric; ``stellsym`` on this
    class controls only the transported surface / mode set of subclasses.
    Seed dofs live on ``seed_surf``; this class does not own a dof vector
    (``SurfaceJAX`` is given an empty placeholder).
    """

    _CHILDREN = (
        'seed_major_radius',
        'seed_minor_radius',
        'quadpoints_phi',
        'quadpoints_theta',
    )
    _AUX = ('nfp', 'stellsym')

    def __init__(
        self,
        nfp,
        stellsym,
        seed_major_radius,
        seed_minor_radius,
        quadpoints_phi,
        quadpoints_theta,
    ):
        self.seed_major_radius = seed_major_radius
        self.seed_minor_radius = seed_minor_radius
        seed_dofs = jnp.asarray([
            seed_major_radius,
            seed_minor_radius,
            seed_minor_radius,
        ])
        # Circular torus is stellarator-symmetric by construction.
        self.seed_surf = SurfaceRZFourierJAX(
            nfp=nfp,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=seed_dofs,
        )
        super().__init__(
            nfp=nfp,
            stellsym=stellsym,
            mpol=1,
            ntor=0,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=jnp.zeros((0,)),
        )

    def _gamma_map_single(self, gamma_init):
        """Map a single seed point ``gamma_init`` of shape ``(3,)`` to ``(3,)``."""
        raise NotImplementedError(
            f'_gamma_map_single is not implemented for {type(self)}'
        )

    @partial(jit, static_argnames=['a', 'b'])
    def gammadash_at_point(self, phi, theta, a: int, b: int) -> jnp.ndarray:
        """Broadcastable ``d^(a+b) gamma / dphi^a dtheta^b`` via map autodiff."""
        def mapped(phi_s, theta_s):
            return self._gamma_map_single(
                self.seed_surf.gammadash_at_point(phi_s, theta_s, 0, 0)
            )

        deriv_fn = mapped
        for _ in range(a):
            deriv_fn = jacfwd(deriv_fn, argnums=0)
        for _ in range(b):
            deriv_fn = jacfwd(deriv_fn, argnums=1)

        phi_b, theta_b = jnp.broadcast_arrays(phi, theta)
        result = vmap(deriv_fn)(phi_b.ravel(), theta_b.ravel())
        return result.reshape(phi_b.shape + (3,))

    def copy_and_set_quadpoints(self, quadpoints_phi, quadpoints_theta):
        children, aux = self.tree_flatten()
        kwargs = {**dict(zip(self._CHILDREN, children)), **aux}
        kwargs['quadpoints_phi'] = quadpoints_phi
        kwargs['quadpoints_theta'] = quadpoints_theta
        return type(self)(**kwargs)

    def gen_winding_surface_dofs(self, *args, **kwargs):
        raise NotImplementedError(
            f'gen_winding_surface_dofs is not yet supported by {type(self)}'
        )

    def gen_winding_surface(self, *args, **kwargs):
        raise NotImplementedError(
            f'gen_winding_surface is not yet supported by {type(self)}'
        )

    @classmethod
    def fit(cls, *args, **kwargs):
        raise NotImplementedError(
            f'fit is not yet supported by {cls}'
        )

    def tree_flatten(self):
        children = tuple(getattr(self, n) for n in self._CHILDREN)
        aux_data = {n: getattr(self, n) for n in self._AUX}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**dict(zip(cls._CHILDREN, children)), **aux_data)


@tree_util.register_pytree_node_class
class SurfaceFlowJAX(SurfaceTransportJAX):
    """Surface transported by integrating an abstract flow with RK4."""

    _AUX = SurfaceTransportJAX._AUX + ('step_num',)

    def __init__(
        self,
        nfp,
        stellsym,
        seed_major_radius,
        seed_minor_radius,
        quadpoints_phi,
        quadpoints_theta,
        step_num,
    ):
        super().__init__(
            nfp=nfp,
            stellsym=stellsym,
            seed_major_radius=seed_major_radius,
            seed_minor_radius=seed_minor_radius,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        self.step_num = step_num

    def _flow_single(self, gamma):
        """Flow field ``dγ/dt`` at a single point ``gamma`` of shape ``(3,)``."""
        raise NotImplementedError(
            f'_flow_single is not yet supported by {type(self)}'
        )

    def _gamma_map_single(self, gamma_init):
        """RK4 integrate ``dγ/dt = _flow_single(γ)`` from ``t=0`` to ``t=1``."""
        dt = 1.0 / self.step_num

        def body(_, y):
            k1 = self._flow_single(y)
            k2 = self._flow_single(y + 0.5 * dt * k1)
            k3 = self._flow_single(y + 0.5 * dt * k2)
            k4 = self._flow_single(y + dt * k3)
            return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return lax.fori_loop(0, self.step_num, body, gamma_init)


@tree_util.register_pytree_node_class
class SurfaceHarmonicJAX(SurfaceFlowJAX):
    """Seed surface transported by the gradient flow of an nfp-periodic harmonic potential.

    The flow at a point ``γ`` is

        v(γ) = gamma_scale · ∇_ĝ Φ(ĝ),   ĝ = γ / gamma_scale,

    where ``Φ`` is the weighted solid-harmonic potential.  Harmonic only ever
    sees the dimensionless ``ĝ``.

    By default ``irregular=True`` (``I_l^m = R_l^m / r^{2l+1}``), which decays
    at infinity and is singular at the origin.  Regular harmonics
    (``irregular=False``) grow like ``r^{l-1}`` in the gradient and can
    radially runaway under RK4.

    Prefer ``backend='recursion'`` for nested AD through RK4.
    ``backend='gemm'`` is regular-only and incompatible with ``irregular=True``.

    Parameters
    ----------
    dofs
        Flat length-``num_modes`` coefficient vector only (no dict).  Defaults
        to zeros.  Use :meth:`dofs_dict` to view them as ``{(l, a): w}``.
    backend
        ``'recursion'`` or ``'gemm'``.
    gamma_scale
        Length scale (traced).  Defaults to ``seed_major_radius``.
    irregular
        Use irregular solid harmonics (default ``True``).
    """

    _CHILDREN = SurfaceFlowJAX._CHILDREN + ('dofs', 'gamma_scale')
    _AUX = SurfaceFlowJAX._AUX + ('a_max', 'l_max', 'backend', 'irregular')

    def __init__(
        self,
        nfp,
        stellsym,
        seed_major_radius,
        seed_minor_radius,
        quadpoints_phi,
        quadpoints_theta,
        step_num,
        a_max,
        l_max,
        dofs=None,
        backend='recursion',
        gamma_scale=None,
        irregular=True,
    ):
        super().__init__(
            nfp=nfp,
            stellsym=stellsym,
            seed_major_radius=seed_major_radius,
            seed_minor_radius=seed_minor_radius,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            step_num=step_num,
        )
        self.a_max = int(a_max)
        self.l_max = int(l_max)
        self.backend = str(backend).lower()
        self.irregular = bool(irregular)
        if self.backend not in _BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(_BACKENDS)}, got {backend!r}"
            )

        self.harmonic = _make_harmonic(
            self.backend,
            int(nfp),
            self.a_max,
            self.l_max,
            bool(stellsym),
            self.irregular,
        )

        if dofs is None:
            dofs = jnp.zeros((self.harmonic.num_modes,))
        else:
            dofs = jnp.asarray(dofs)
        if dofs.shape != (self.harmonic.num_modes,):
            raise ValueError(
                f'dofs has the wrong shape {tuple(dofs.shape)}; '
                f'expected {(self.harmonic.num_modes,)}'
            )
        self.dofs = dofs

        if gamma_scale is None:
            self.gamma_scale = jnp.asarray(seed_major_radius)
        else:
            self.gamma_scale = jnp.asarray(gamma_scale)

    def dofs_dict(self):
        """Return harmonic coefficients as ``{(l, a): weight}`` via ``unravel``."""
        return self.harmonic.unravel(self.dofs)

    def _flow_single(self, gamma):
        """``v = gamma_scale * ∇Φ(gamma / gamma_scale)`` with Φ the harmonic potential."""
        g = gamma / self.gamma_scale
        return self.gamma_scale * self.harmonic.grad_run_weighted_dofs(
            g, self.dofs
        )


# ---------------------------------------------------------------------------
# Cylindrical shear transport
# ---------------------------------------------------------------------------

def _half_range_cos(x, x_min, x_max, m_max):
    """Half-range cosine basis on ``[x_min, x_max]``, length ``m_max + 1``.

    Uses ``cos(k * pi * u)`` with ``u = (x - x_min) / (x_max - x_min)`` for
    ``k = 0 .. m_max``.  Unlike ``cos(k * pi * x / x_max)`` this spans
    arbitrary (not just even) functions on the interval.
    """
    u = (x - x_min) / (x_max - x_min)
    return jnp.cos(jnp.pi * u * jnp.arange(m_max + 1))


def _phi_basis(phi, nfp, m_max, stellsym):
    """nfp-periodic Fourier basis in normalized toroidal angle ``phi``.

    Returns length ``m_max + 1`` when ``stellsym`` (cosines only, including
    the constant ``n=0``), or ``2 * m_max + 1`` otherwise
    (``cos(0..M)`` then ``sin(1..M)``; ``sin(0)`` is identically zero and
    never stored).
    """
    n = jnp.arange(m_max + 1)
    angles = 2.0 * jnp.pi * nfp * n * phi
    cos_part = jnp.cos(angles)
    if stellsym:
        return cos_part
    # Skip sin(n=0) which is always zero.
    sin_part = jnp.sin(angles[1:])
    return jnp.concatenate([cos_part, sin_part])


def _axis_len(m_max, is_phi, stellsym):
    """Number of basis functions for one axis given max mode number."""
    if is_phi and not stellsym:
        return 2 * m_max + 1
    return m_max + 1


def _build_shear_plan(m_shear, stellsym):
    """Build static per-stage dof layout from ``m_shear``.

    Returns a tuple of ``(kind, len_a, len_b, offset, size)`` where
    ``kind`` is 0/1/2 for A/B/C, ``size = len_a * len_b - 1`` (no constant),
    and ``offset`` is the start index into the flat dof vector.
    """
    plan = []
    offset = 0
    for i, (ma, mb) in enumerate(m_shear):
        kind = i % 3
        # kind 0 (A): axis a = phi, axis b = z
        # kind 1 (B): axis a = z,   axis b = r
        # kind 2 (C): axis a = r,   axis b = phi
        if kind == 0:
            len_a = _axis_len(ma, is_phi=True, stellsym=stellsym)
            len_b = _axis_len(mb, is_phi=False, stellsym=stellsym)
        elif kind == 1:
            len_a = _axis_len(ma, is_phi=False, stellsym=stellsym)
            len_b = _axis_len(mb, is_phi=False, stellsym=stellsym)
        else:
            len_a = _axis_len(ma, is_phi=False, stellsym=stellsym)
            len_b = _axis_len(mb, is_phi=True, stellsym=stellsym)
        size = len_a * len_b - 1
        plan.append((kind, len_a, len_b, offset, size))
        offset += size
    return tuple(plan), offset


@tree_util.register_pytree_node_class
class SurfaceShearJAX(SurfaceTransportJAX):
    """Seed surface transported by a sequence of cylindrical Fourier shears.

    Stages cycle ``A -> B -> C -> A -> ...`` for ``n_steps = len(m_shear)``:

    - ``A_i``: ``r  -> sqrt(r^2 + 2 f_i(phi, z))``
    - ``B_i``: ``phi -> phi + g_i(z, r)``  (normalized turns)
    - ``C_i``: ``z  -> z + h_i(r, phi)``

    Each of ``f, g, h`` is a tensor-product series with no constant term.
    Phi dependence is nfp-periodic (and cosine-only when ``stellsym``);
    z and r use half-range cosines on ``[-z_max, z_max]`` and ``[0, r_max]``.

    ``m_shear[i]`` gives the two max mode numbers for stage ``i`` (axis order
    matches the series arguments above).  Flat ``dofs`` stores all stage
    coefficients row-major with the leading ``(0,0)`` entry of each stage
    omitted.

    Warning: ``A_i`` yields NaN wherever ``r^2 + 2 f_i < 0`` (no clamping).
    """

    _CHILDREN = SurfaceTransportJAX._CHILDREN + ('dofs', 'z_max', 'r_max')
    _AUX = SurfaceTransportJAX._AUX + ('m_shear',)

    def __init__(
        self,
        nfp,
        stellsym,
        seed_major_radius,
        seed_minor_radius,
        quadpoints_phi,
        quadpoints_theta,
        m_shear,
        dofs=None,
        z_max=None,
        r_max=None,
    ):
        super().__init__(
            nfp=nfp,
            stellsym=stellsym,
            seed_major_radius=seed_major_radius,
            seed_minor_radius=seed_minor_radius,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

        # Normalize m_shear to a hashable tuple of int pairs (jit aux data).
        m_shear = tuple(
            (int(row[0]), int(row[1])) for row in m_shear
        )
        if not m_shear:
            raise ValueError('m_shear must contain at least one stage')
        for i, (ma, mb) in enumerate(m_shear):
            if ma < 0 or mb < 0:
                raise ValueError(
                    f'm_shear[{i}] = {(ma, mb)} has a negative mode number'
                )
        self.m_shear = m_shear

        self._plan, self.num_modes = _build_shear_plan(
            self.m_shear, bool(stellsym)
        )

        if dofs is None:
            dofs = jnp.zeros((self.num_modes,))
        else:
            dofs = jnp.asarray(dofs)
        if dofs.shape != (self.num_modes,):
            raise ValueError(
                f'dofs has the wrong shape {tuple(dofs.shape)}; '
                f'expected {(self.num_modes,)}'
            )
        self.dofs = dofs

        if z_max is None:
            self.z_max = jnp.asarray(2.0 * seed_minor_radius)
        else:
            self.z_max = jnp.asarray(z_max)
        if r_max is None:
            self.r_max = jnp.asarray(
                seed_major_radius + 2.0 * seed_minor_radius
            )
        else:
            self.r_max = jnp.asarray(r_max)

    def dofs_matrices(self):
        """Unpack flat dofs into per-stage coefficient matrices (with C00=0)."""
        return self._coeff_matrices(self.dofs)

    def get_bounds(
        self,
        lbf=0.0,
        ubf=None,
        lbg=-0.1,
        ubg=0.1,
        lbh=None,
        ubh=None,
    ):
        """Return ``(lb, ub)`` bound arrays with the same shape as ``dofs``.

        Stage kinds map to bound pairs: A/f -> ``(lbf, ubf)``, B/g ->
        ``(lbg, ubg)``, C/h -> ``(lbh, ubh)``.  Defaults: ``ubf = r_max``,
        ``lbh = -z_max``, ``ubh = z_max`` (``None`` resolves to those).
        """
        if ubf is None:
            ubf = self.r_max
        if lbh is None:
            lbh = -self.z_max
        if ubh is None:
            ubh = self.z_max

        dtype = self.dofs.dtype
        lbs = []
        ubs = []
        for kind, _, _, _, size in self._plan:
            if kind == 0:
                lo, hi = lbf, ubf
            elif kind == 1:
                lo, hi = lbg, ubg
            else:
                lo, hi = lbh, ubh
            lbs.append(jnp.full((size,), lo, dtype=dtype))
            ubs.append(jnp.full((size,), hi, dtype=dtype))
        return jnp.concatenate(lbs), jnp.concatenate(ubs)

    def _coeff_matrices(self, dofs):
        """Split the flat dof vector into dense stage matrices.

        Each stage stores ``len_a * len_b - 1`` coefficients: row-major over
        ``(axis-a mode, axis-b mode)`` with the leading ``(0, 0)`` constant
        term omitted.  Re-insert that constant as an explicit zero so the
        dense matrix product ``basis_a @ C @ basis_b`` still works.
        """
        mats = []
        for _, len_a, len_b, off, size in self._plan:
            block = jnp.concatenate([
                jnp.zeros((1,), dtype=dofs.dtype),
                dofs[off:off + size],
            ])
            mats.append(block.reshape(len_a, len_b))
        return mats

    def _eval_series(self, basis_a, C, basis_b):
        """Scalar ``basis_a @ C @ basis_b`` for 1-D basis vectors."""
        return basis_a @ (C @ basis_b)

    def _gamma_map_single(self, gamma_init):
        """Apply the A/B/C shear sequence to one Cartesian point ``(3,)``."""
        x, y, z = gamma_init[0], gamma_init[1], gamma_init[2]
        r = jnp.sqrt(x * x + y * y)
        # Normalized toroidal angle in turns (quadcoil convention).
        phi = jnp.arctan2(y, x) / (2.0 * jnp.pi)

        mats = self._coeff_matrices(self.dofs)
        z_max = self.z_max
        r_max = self.r_max
        nfp = self.nfp
        stellsym = self.stellsym

        # Fully unrolled Python loop: every stage shape is static.
        for (kind, _, _, _, _), C, (ma, mb) in zip(
            self._plan, mats, self.m_shear
        ):
            if kind == 0:
                # A: r -> sqrt(r^2 + 2 f(phi, z))
                # m_shear = (max phi, max z)
                P = _phi_basis(phi, nfp, ma, stellsym)
                Z = _half_range_cos(z, -z_max, z_max, mb)
                f = self._eval_series(P, C, Z)
                r = jnp.sqrt(r * r + 2.0 * f)
            elif kind == 1:
                # B: phi -> phi + g(z, r)  (normalized turns)
                # m_shear = (max z, max r)
                Z = _half_range_cos(z, -z_max, z_max, ma)
                R = _half_range_cos(r, 0.0, r_max, mb)
                g = self._eval_series(Z, C, R)
                phi = phi + g
            else:
                # C: z -> z + h(r, phi)
                # m_shear = (max r, max phi)
                R = _half_range_cos(r, 0.0, r_max, ma)
                P = _phi_basis(phi, nfp, mb, stellsym)
                h = self._eval_series(R, C, P)
                z = z + h

        # Back to Cartesian.  No phi wrapping needed: the basis is exactly
        # period-1/nfp, so arctan2's branch cut is invisible to the series.
        ang = 2.0 * jnp.pi * phi
        return jnp.stack([r * jnp.cos(ang), r * jnp.sin(ang), z])


# ---------------------------------------------------------------------------
# Beltrami eigenfunction transport
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _make_beltrami_basis(
    m_per_period,
    nfp,
    min_lam,
    max_lam,
    n_lam,
    radius,
    max_order,
    max_iter,
    tol,
    pi_guard,
    polish,
):
    """Memoized BeltramiBasis (static aux data only; stellsym forced True)."""
    from beltrami import BeltramiBasis

    return BeltramiBasis(
        m_per_period=int(m_per_period),
        nfp=int(nfp),
        min_lam=float(min_lam),
        max_lam=float(max_lam),
        n_lam=int(n_lam),
        radius=float(radius),
        max_order=int(max_order),
        max_iter=int(max_iter),
        tol=float(tol),
        pi_guard=float(pi_guard),
        polish=bool(polish),
        stellsym=True,
    )


@tree_util.register_pytree_node_class
class SurfaceBeltramiJAX(SurfaceFlowJAX):
    """Seed surface transported by a stellarator-symmetric Beltrami flow.

    The physical cylinder of radius ``box_r`` and height ``box_z`` is mapped
    to the Morse reference cylinder (unit height, radius ``a = box_r/box_z``)
    by the uniform scale ``γ → γ / box_z``.  The flow is

        v(γ) = box_z · B(γ / box_z),

    where ``B`` is the weighted sum of Beltrami eigenmodes.  The leading
    ``box_z`` keeps dof magnitudes independent of box size (mirrors
    ``SurfaceHarmonicJAX.gamma_scale``).

    ``stellsym=False`` is not supported: complex Beltrami callables cannot
    transport a real surface.  Raise ``NotImplementedError`` and force
    ``stellsym=True`` on the underlying :class:`~beltrami.BeltramiBasis`.

    ``box_r`` / ``box_z`` are aux (not children): the NumPy/SciPy root search
    needs a concrete radius, so they cannot be traced.

    Parameters
    ----------
    dofs
        Flat length-``len(basis)`` coefficient vector.  Defaults to zeros.
    box_r, box_z
        Physical cylinder radius and height.  Defaults
        ``seed_major_radius + 2*seed_minor_radius`` and
        ``2*seed_minor_radius``.
    m_per_period, min_lam, max_lam, n_lam, max_order, max_iter, tol,
    pi_guard, polish
        Forwarded to :class:`~beltrami.BeltramiBasis` (with ``nfp`` from
        this surface and ``radius = box_r / box_z``).
    """

    _CHILDREN = SurfaceFlowJAX._CHILDREN + ('dofs',)
    _AUX = SurfaceFlowJAX._AUX + (
        'box_r',
        'box_z',
        'm_per_period',
        'min_lam',
        'max_lam',
        'n_lam',
        'max_order',
        'max_iter',
        'tol',
        'pi_guard',
        'polish',
    )

    def __init__(
        self,
        nfp,
        stellsym,
        seed_major_radius,
        seed_minor_radius,
        quadpoints_phi,
        quadpoints_theta,
        step_num,
        m_per_period,
        min_lam,
        max_lam,
        n_lam,
        dofs=None,
        box_r=None,
        box_z=None,
        max_order=50,
        max_iter=100,
        tol=1e-5,
        pi_guard=1e-3,
        polish=True,
    ):
        if not stellsym:
            raise NotImplementedError(
                "SurfaceBeltramiJAX requires stellsym=True "
                "(complex Beltrami fields cannot transport a real surface)"
            )
        super().__init__(
            nfp=nfp,
            stellsym=stellsym,
            seed_major_radius=seed_major_radius,
            seed_minor_radius=seed_minor_radius,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            step_num=step_num,
        )
        self.m_per_period = int(m_per_period)
        self.min_lam = float(min_lam)
        self.max_lam = float(max_lam)
        self.n_lam = int(n_lam)
        self.max_order = int(max_order)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.pi_guard = float(pi_guard)
        self.polish = bool(polish)

        if box_r is None:
            box_r = float(seed_major_radius) + 2.0 * float(seed_minor_radius)
        if box_z is None:
            box_z = 2.0 * float(seed_minor_radius)
        self.box_r = float(box_r)
        self.box_z = float(box_z)
        if not (self.box_r > 0.0 and self.box_z > 0.0):
            raise ValueError(
                f"box_r and box_z must be > 0, got box_r={self.box_r!r}, "
                f"box_z={self.box_z!r}"
            )

        radius = self.box_r / self.box_z
        self.basis = _make_beltrami_basis(
            self.m_per_period,
            int(nfp),
            self.min_lam,
            self.max_lam,
            self.n_lam,
            radius,
            self.max_order,
            self.max_iter,
            self.tol,
            self.pi_guard,
            self.polish,
        )

        n_modes = len(self.basis)
        if dofs is None:
            dofs = jnp.zeros((n_modes,))
        else:
            dofs = jnp.asarray(dofs)
        if dofs.shape != (n_modes,):
            raise ValueError(
                f'dofs has the wrong shape {tuple(dofs.shape)}; '
                f'expected {(n_modes,)}'
            )
        self.dofs = dofs

    def _flow_single(self, gamma):
        """``v = box_z * B(gamma / box_z)`` with B the Beltrami weighted sum."""
        g = gamma / self.box_z
        r = jnp.sqrt(g[0] * g[0] + g[1] * g[1])
        # Beltrami callables use radians (exp(1j m phi)), not normalized turns.
        phi = jnp.arctan2(g[1], g[0])
        return self.box_z * self.basis.eval(r, phi, g[2], self.dofs)
