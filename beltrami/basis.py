"""Multi-m Beltrami eigenfunction basis for a finite cylinder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .beltrami import (
    find_beltrami_lam,
    find_beltrami_lam_axisym,
    generate_beltrami_callable,
    generate_beltrami_weighted_callable,
)

# Measured peak intermediate cost in ``_jax_radial_rho`` (~1320 bytes per
# field point per retained Fourier term). Used to size evaluation chunks.
# Batched eval scales as M·N, so the chunk divisor multiplies by block size.
_BYTES_PER_POINT_PER_ORDER = 1320
_EVAL_MEMORY_BUDGET = 2 * 1024**3  # 2 GiB
# Floor on the innermost radial sample so JAX ``bessel_jn`` stays away from
# its NaN region near the axis (arguments below ~2e-5).
_R_GRID_FLOOR_FRAC = 1e-4


class BeltramiBasis:
    """Flattened Beltrami eigenfunction basis across several azimuthal modes.

    For each ``m`` in ``np.arange(0, m_per_period + 1) * nfp``, runs the
    appropriate spectrum finder (``find_beltrami_lam_axisym`` for ``m = 0``,
    ``find_beltrami_lam`` for ``m >= 1``) then builds both a per-mode callable
    list and one weighted-sum callable per ``m`` block. Mode count per ``m``
    is data-dependent, so the flattened length is not a simple product.

    With ``stellsym=True`` every callable returns ``float64`` ``Re(B)``,
    which is itself a Beltrami field and is anti-equivariant under the
    180° rotation about the horizontal axis at ``φ = π/nfp``
    (``B(Qx) = −Q B(x)``, cylindrical parity ``(B_R, B_φ, B_Z) = (−, +, +)``).
    Axisymmetric even-``q`` modes occupy the ``an`` slot and are purely
    imaginary, so they are dropped before the weighted callable is built.

    Attributes
    ----------
    beltrami_basis
        One jitted ``(r, phi, z) -> B_xyz`` callable per accepted eigenmode.
        ``complex128`` when ``stellsym`` is false; ``float64`` when true.
    m
        Azimuthal mode number for each basis function, shape ``(n_basis,)``.
    lam
        Eigenvalue for each basis function, shape ``(n_basis,)``.
    stellsym
        Whether the basis is restricted to the stellarator-symmetric
        (anti-equivariant) real quadrature.
    """

    def __init__(
        self,
        m_per_period: int,
        nfp: int,
        min_lam: float,
        max_lam: float,
        n_lam: int,
        radius: float,
        max_order: int = 50,
        max_iter: int = 100,
        tol: float = 1e-5,
        pi_guard: float = 1e-3,
        polish: bool = True,
        stellsym: bool = False,
    ) -> None:
        if not isinstance(m_per_period, (int, np.integer)) or int(m_per_period) < 1:
            raise ValueError(
                f"m_per_period must be an integer >= 1, got {m_per_period!r}"
            )
        if not isinstance(nfp, (int, np.integer)) or int(nfp) < 1:
            raise ValueError(f"nfp must be an integer >= 1, got {nfp!r}")
        if not (radius > 0.0):
            raise ValueError(f"radius must be > 0, got {radius!r}")
        if not (min_lam > 0.0 and max_lam > min_lam):
            raise ValueError(
                f"need 0 < min_lam < max_lam, got min_lam={min_lam!r}, max_lam={max_lam!r}"
            )

        self.m_per_period = int(m_per_period)
        self.nfp = int(nfp)
        self.min_lam = float(min_lam)
        self.max_lam = float(max_lam)
        self.n_lam = int(n_lam)
        self.radius = float(radius)
        self.max_order = int(max_order)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.pi_guard = float(pi_guard)
        self.polish = bool(polish)
        self.stellsym = bool(stellsym)

        m_list = np.arange(0, self.m_per_period + 1, dtype=int) * self.nfp

        callables: list[Callable] = []
        m_vals: list[NDArray[np.integer]] = []
        lam_vals: list[NDArray[np.floating]] = []
        # Per-m blocks for batched eval: (offset, n_modes, weighted_fn).
        blocks: list[tuple[int, int, Callable]] = []
        offset = 0

        for m in m_list:
            if int(m) == 0:
                spectrum = find_beltrami_lam_axisym(
                    min_lam=self.min_lam,
                    max_lam=self.max_lam,
                    radius=self.radius,
                    max_order=self.max_order,
                )
            else:
                spectrum = find_beltrami_lam(
                    m=int(m),
                    min_lam=self.min_lam,
                    max_lam=self.max_lam,
                    n_lam=self.n_lam,
                    max_order=self.max_order,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    radius=self.radius,
                    pi_guard=self.pi_guard,
                    polish=self.polish,
                )
            # Filter before building the weighted callable (it contracts
            # internally, so post-hoc list masking no longer works).
            if self.stellsym and int(m) == 0 and spectrum.n_modes > 0:
                keep = ~np.any(spectrum.an != 0.0, axis=1)
                spectrum = replace(
                    spectrum,
                    lam=np.asarray(spectrum.lam[keep], dtype=float),
                    residual=np.asarray(spectrum.residual[keep], dtype=float),
                    an=np.asarray(spectrum.an[keep], dtype=float),
                    bn=np.asarray(spectrum.bn[keep], dtype=float),
                    kr2_even=np.asarray(spectrum.kr2_even[keep], dtype=float),
                    kr2_odd=np.asarray(spectrum.kr2_odd[keep], dtype=float),
                    min_bessel_denom=np.asarray(
                        spectrum.min_bessel_denom[keep], dtype=float
                    ),
                    # Axisym status is length n_modes (one ROOT per mode).
                    status=np.asarray(spectrum.status[keep]),
                )

            n_modes = spectrum.n_modes
            if n_modes == 0:
                continue

            fns = generate_beltrami_callable(spectrum, real=self.stellsym)
            weighted = generate_beltrami_weighted_callable(
                spectrum, real=self.stellsym
            )
            callables.extend(fns)
            m_vals.append(np.full(n_modes, int(m), dtype=int))
            lam_vals.append(np.asarray(spectrum.lam, dtype=float))
            blocks.append((offset, n_modes, weighted))
            offset += n_modes

        self.beltrami_basis: list[Callable] = callables
        self._blocks = blocks
        if m_vals:
            self.m: NDArray[np.integer] = np.concatenate(m_vals)
            self.lam: NDArray[np.floating] = np.concatenate(lam_vals)
        else:
            self.m = np.empty(0, dtype=int)
            self.lam = np.empty(0, dtype=float)

        # Prebuild the jitted eval closure (self is not a pytree).
        self._eval_fn = self._make_eval_fn(blocks)

    @staticmethod
    def _make_eval_fn(blocks: list[tuple[int, int, Callable]]):
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
        # Freeze blocks into the closure so jit does not capture ``self``.
        frozen = tuple(blocks)

        def _eval(r, phi, z, weights):
            total = None
            for off, n_modes, fn in frozen:
                w = weights[off : off + n_modes]
                contrib = fn(r, phi, z, w)
                total = contrib if total is None else total + contrib
            if total is None:
                r0, _, _ = jnp.broadcast_arrays(
                    jnp.asarray(r, dtype=jnp.float64),
                    jnp.asarray(phi, dtype=jnp.float64),
                    jnp.asarray(z, dtype=jnp.float64),
                )
                return jnp.zeros(r0.shape + (3,), dtype=jnp.float64)
            return total

        return jax.jit(_eval)

    def __len__(self) -> int:
        return len(self.beltrami_basis)

    def eval(self, r, phi, z, weights):
        """Weighted sum of all basis fields at ``(r, phi, z)``.

        Parameters
        ----------
        r, phi, z
            Broadcastable cylindrical coordinates (``phi`` in radians).
        weights
            Length-``len(self)`` coefficient vector. Shape is checked at
            trace time (static under jit), so a mismatch raises.

        Returns
        -------
        B_xyz
            Shape ``(..., 3)``. ``float64`` when ``stellsym``; else complex.
        """
        import jax.numpy as jnp

        weights = jnp.asarray(weights)
        n = len(self)
        if weights.shape[0] != n:
            raise ValueError(
                f"weights has length {weights.shape[0]}; expected {n}"
            )
        return self._eval_fn(r, phi, z, weights)

    def visualize(
        self,
        weights,
        name: str,
        n_r: int = 32,
        n_phi: int = 32,
        n_z: int = 32,
    ) -> Path:
        """Evaluate the weighted field on a cylindrical grid and write a ``.vts``.

        Axis inclusion follows the active modes: ``r = 0`` is included only
        when every nonzero-weight mode has ``m = 0``. Chunk size accounts for
        the batched ``M·N`` intermediate cost.

        Parameters
        ----------
        weights
            Length-``len(self)`` coefficient vector.
        name
            Filename stem. The written file is ``{name}.vts``.
        n_r, n_phi, n_z
            Grid resolution (defaults ``32``). The azimuthal grid covers one
            field period, ``0 <= phi <= 2*pi/nfp``.

        Returns
        -------
        Path
            Path of the written ``.vts`` file (as reported by ``gridToVTK``).
        """
        try:
            from pyevtk.hl import gridToVTK
        except ImportError as exc:
            raise ImportError(
                "pyevtk must be installed to save vtk files."
            ) from exc

        weights_arr = np.asarray(weights, dtype=float).reshape(-1)
        if weights_arr.shape[0] != len(self):
            raise ValueError(
                f"weights has length {weights_arr.shape[0]}; expected {len(self)}"
            )
        if not (isinstance(n_r, (int, np.integer)) and int(n_r) >= 2):
            raise ValueError(f"n_r must be an integer >= 2, got {n_r!r}")
        if not (isinstance(n_phi, (int, np.integer)) and int(n_phi) >= 2):
            raise ValueError(f"n_phi must be an integer >= 2, got {n_phi!r}")
        if not (isinstance(n_z, (int, np.integer)) and int(n_z) >= 2):
            raise ValueError(f"n_z must be an integer >= 2, got {n_z!r}")
        n_r, n_phi, n_z = int(n_r), int(n_phi), int(n_z)

        radius = self.radius
        active = np.asarray(weights_arr) != 0.0
        include_axis = (not np.any(active)) or (
            not np.any(self.m[active] > 0)
        )
        if include_axis:
            r = np.linspace(0.0, radius, n_r)
        else:
            # Drop the axis sample, then floor the innermost radius so large
            # n_r cannot re-enter the JAX bessel_jn NaN region near r = 0.
            r = np.linspace(0.0, radius, n_r + 1)[1:]
            r_floor = _R_GRID_FLOOR_FRAC * radius
            r = np.maximum(r, r_floor)
        # Export one nfp-periodic wedge. Both endpoints are retained as the
        # two periodic boundaries at phi = 0 and 2π/nfp.
        phi = np.linspace(0.0, 2.0 * np.pi / self.nfp, n_phi)
        z = np.linspace(-0.5, 0.5, n_z)

        R, Phi, Z = np.meshgrid(r, phi, z, indexing="ij")
        flat_r = R.ravel()
        flat_phi = Phi.ravel()
        flat_z = Z.ravel()
        n_pts = flat_r.size

        # Batched intermediates scale as M·N; size chunks accordingly.
        m_max = max((n for _, n, _ in self._blocks), default=1)
        chunk = max(
            1,
            int(
                _EVAL_MEMORY_BUDGET
                / (_BYTES_PER_POINT_PER_ORDER * m_max * self.max_order)
            ),
        )
        out_dtype = np.float64 if self.stellsym else np.complex128
        out = np.empty((n_pts, 3), dtype=out_dtype)
        for start in range(0, n_pts, chunk):
            sl = slice(start, start + chunk)
            out[sl] = np.asarray(
                self.eval(flat_r[sl], flat_phi[sl], flat_z[sl], weights_arr)
            )

        # Cartesian mesh and Cartesian field components.
        x = np.ascontiguousarray((R * np.cos(Phi)).astype(np.float64))
        y = np.ascontiguousarray((R * np.sin(Phi)).astype(np.float64))
        z_grid = np.ascontiguousarray(Z.astype(np.float64))

        b = out.reshape(n_r, n_phi, n_z, 3)
        b_magnitude = np.ascontiguousarray(
            np.linalg.norm(b, axis=-1).astype(np.float64)
        )
        if self.stellsym:
            point_data = {
                "B": (
                    np.ascontiguousarray(b[..., 0].astype(np.float64)),
                    np.ascontiguousarray(b[..., 1].astype(np.float64)),
                    np.ascontiguousarray(b[..., 2].astype(np.float64)),
                ),
                "B_magnitude": b_magnitude,
            }
        else:
            point_data = {
                "B_real": (
                    np.ascontiguousarray(b[..., 0].real),
                    np.ascontiguousarray(b[..., 1].real),
                    np.ascontiguousarray(b[..., 2].real),
                ),
                "B_imag": (
                    np.ascontiguousarray(b[..., 0].imag),
                    np.ascontiguousarray(b[..., 1].imag),
                    np.ascontiguousarray(b[..., 2].imag),
                ),
                "B_magnitude": b_magnitude,
            }

        written = gridToVTK(
            name,
            x,
            y,
            z_grid,
            pointData=point_data,
        )
        return Path(written)

    def visualize_basis(
        self,
        i: int,
        name: str,
        n_r: int = 32,
        n_phi: int = 32,
        n_z: int = 32,
    ) -> Path:
        """Evaluate basis function ``i`` on a cylindrical grid and write a ``.vts``.

        Thin wrapper around :meth:`visualize` with one-hot weights. The
        written file is ``{name}_i{i}_m{m}_lam{round(lam, 3)}.vts``.

        For ``m >= 1`` the radial grid excludes the degenerate axis (``r = 0``),
        where the cylindrical ``1/r`` terms in the CK formula are delicate.
        For ``m = 0`` the axis is included — the mode peaks there and the
        Bessel evaluators are finite after the ``_jax_jm_djm`` small-z patch.

        When ``stellsym`` is false the exported arrays are the complex
        Cartesian field split into ``B_real``, ``B_imag``, and
        ``B_magnitude``. When ``stellsym`` is true the callable is already
        real, so the file carries a single ``B`` vector plus
        ``B_magnitude``.
        """
        if not isinstance(i, (int, np.integer)):
            raise TypeError(f"i must be an integer, got {type(i)!r}")
        i = int(i)
        if i < 0:
            i += len(self)
        if not 0 <= i < len(self):
            raise IndexError(
                f"basis index {i} out of range for n_basis={len(self)}"
            )
        m_i = int(self.m[i])
        lam_i = float(self.lam[i])
        weights = np.zeros(len(self), dtype=float)
        weights[i] = 1.0
        stem = f"{name}_i{i}_m{m_i}_lam{round(lam_i, 3)}"
        return self.visualize(weights, stem, n_r=n_r, n_phi=n_phi, n_z=n_z)
