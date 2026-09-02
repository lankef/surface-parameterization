"""Periodic real solid harmonics for an nfp-periodic potential.

Racah-normalised real solid harmonics matching the convention verified
against ``scipy.special.sph_harm_y`` in ``harmonics/verify.py``.

Regular (``R_l^{\\pm m}``) via Cartesian recursion:

* Diagonal seed
      R[m,  m] = c_m * Re[(x + iy)^m],
      R[m, -m] = c_m * Im[(x + iy)^m],
  with ``c_m = sqrt(2 * (2m-1)!! / (2m)!!)`` for ``m > 0``.
* Vertical step at fixed ``m``
      R[l, m] = ((2l-1) z R[l-1, m]
                 - sqrt((l-1)^2 - m^2) r^2 R[l-2, m]) / sqrt(l^2 - m^2)
  (second term dropped when ``l - 1 == m``).

Irregular (default on ``HarmonicRecursion``):
      I_l^m = R_l^m / r^{2l+1}
  (same angular structure; singular at the origin; preferred for bounded
  surface transport because ``|∇I|`` decays at infinity).

Modes are restricted to orders ``m = nfp * |a|`` with keys ``(l, a)``:

* ``a > 0``  -- cosine-type  (``R[l, +nfp*a]``)
* ``a < 0``  -- sine-type    (``R[l, -nfp*|a|]``)
* ``a = 0``  -- zonal        (``R[l, 0]``), always included for
  ``1 <= l <= l_max`` (the constant ``R[0, 0]`` is an internal seed only)

Mode set::

    {(l, 0) : 1 <= l <= l_max}
    union
    {(l, a) : 1 <= |a| <= a_max, nfp*|a| <= l <= l_max,
              a > 0 or not stellsym}

Canonical order: ``|a|`` ascending, then ``+a`` before ``-a``, then ``l``
ascending.

Two backends share the same API:

* ``HarmonicRecursion`` -- Cartesian recursion (reference; near machine
  precision). Supports ``irregular=True`` (default). Prefer this for
  single-point flow / nested surface AD; gradients use ``jacfwd``.
* ``HarmonicGEMM`` -- factored monomial form for batch ``run_all``
  (regular only). Spatial gradients also use ``jacfwd``.

Note: SurfaceJAX geometry requests several ``gammadash(a, b)`` orders as
independent jitted calls, each re-integrating the full RK4 flow. A single
Taylor-mode / jet pass emitting all needed orders would remove that
redundancy, but requires a base-class API change.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

Key = Tuple[int, int]  # (l, a)
Weights = Mapping[Key, object]

__all__ = ["Harmonic", "HarmonicRecursion", "HarmonicGEMM"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _diag_const(m: int) -> float:
    """c_m = sqrt(2 * (2m-1)!! / (2m)!!) for m > 0; c_0 = 1."""
    if m == 0:
        return 1.0
    c = 2.0
    for j in range(1, m + 1):
        c *= (2 * j - 1) / (2 * j)
    return float(np.sqrt(c))


def _build_keys(nfp: int, a_max: int, l_max: int, stellsym: bool) -> Tuple[Key, ...]:
    keys = []
    for l in range(1, l_max + 1):
        keys.append((l, 0))
    for a_abs in range(1, a_max + 1):
        m = nfp * a_abs
        if m > l_max:
            break
        signs = (1,) if stellsym else (1, -1)
        for s in signs:
            a = s * a_abs
            for l in range(m, l_max + 1):
                keys.append((l, a))
    return tuple(keys)


def _poly_mul(p: dict, mono: Tuple[int, int], s: float = 1.0) -> dict:
    """Multiply a {(p, q): coef} polynomial by s * z^{mono[0]} (r^2)^{mono[1]}."""
    out: dict = {}
    for e, c in p.items():
        k = (e[0] + mono[0], e[1] + mono[1])
        out[k] = out.get(k, 0.0) + s * c
    return out


def _poly_add(p: dict, q: dict) -> dict:
    out = dict(p)
    for e, c in q.items():
        out[e] = out.get(e, 0.0) + c
    return out


def _q_poly_recursion(l_max: int, ms: Sequence[int]) -> Dict[Tuple[int, int], dict]:
    """Exact polynomial recursion for Q[l, m](z, r^2) with Q[m, m] = 1."""
    Q: Dict[Tuple[int, int], dict] = {}
    for m in ms:
        Q[(m, m)] = {(0, 0): 1.0}
        for l in range(m + 1, l_max + 1):
            den = np.sqrt(l * l - m * m)
            a = (2 * l - 1) / den
            t = _poly_mul(Q[(l - 1, m)], (1, 0), a)
            if l - 1 > m:
                b = np.sqrt((l - 1) ** 2 - m * m) / den
                t = _poly_add(t, _poly_mul(Q[(l - 2, m)], (0, 1), -b))
            Q[(l, m)] = t
    return Q


# ---------------------------------------------------------------------------
# abstract base
# ---------------------------------------------------------------------------

class Harmonic(abc.ABC):
    """Abstract base for nfp-periodic solid-harmonic evaluators.

    Class attribute ``stellsym`` (default ``True``) selects cosine-only
    modes.  Pass ``stellsym=`` to the constructor to override per instance.

    Hot path for surface transport: ``run_weighted_dofs`` /
    ``grad_run_weighted_dofs`` on a single point of shape ``(3,)``.
    """

    stellsym: bool = True

    def __init__(
        self,
        nfp: int,
        a_max: int,
        l_max: int,
        stellsym: Optional[bool] = None,
    ):
        if nfp < 1:
            raise ValueError(f"nfp must be >= 1, got {nfp}")
        if a_max < 1:
            raise ValueError(f"a_max must be >= 1, got {a_max}")
        if l_max < nfp:
            raise ValueError(f"l_max must be >= nfp, got l_max={l_max}, nfp={nfp}")

        self.nfp = int(nfp)
        self.a_max = int(a_max)
        self.l_max = int(l_max)
        if stellsym is not None:
            self.stellsym = bool(stellsym)

        self.keys: Tuple[Key, ...] = _build_keys(
            self.nfp, self.a_max, self.l_max, self.stellsym
        )
        self._index: Dict[Key, int] = {key: i for i, key in enumerate(self.keys)}

        self._a_abs_list: Tuple[int, ...] = tuple(
            a for a in range(0, self.a_max + 1) if self.nfp * a <= self.l_max
        )
        self._ms: Tuple[int, ...] = tuple(self.nfp * a for a in self._a_abs_list)
        self._c_m: Dict[int, float] = {m: _diag_const(m) for m in self._ms}

        # Lazy: jax.jit does not trace at construction.
        self._run_all_jit = jax.jit(self._run_all_impl)
        self._phi_dofs_single = jax.jit(self._phi_dofs_single_impl)
        self._grad_phi_dofs_single = jax.jit(self._grad_phi_dofs_single_impl)

    def _signs(self, a_abs: int) -> Tuple[int, ...]:
        """Azimuthal signs for family ``|a| = a_abs`` (zonal has no sine)."""
        return (1,) if (a_abs == 0 or self.stellsym) else (1, -1)

    @property
    def num_modes(self) -> int:
        return len(self.keys)

    @staticmethod
    def _unpack_gamma(gamma):
        """Split ``gamma[..., 3]`` into ``(shape, x, y, z)`` with ``x,y,z`` flat."""
        gamma = jnp.asarray(gamma)
        if gamma.shape[-1] != 3:
            raise ValueError(
                f"gamma must have shape (..., 3), got {tuple(gamma.shape)}"
            )
        shape = gamma.shape[:-1]
        flat = gamma.reshape((-1, 3))
        return shape, flat[:, 0], flat[:, 1], flat[:, 2]

    def _angular_pairs(self, x, y) -> List[Tuple]:
        """Re/Im[(x+iy)^{nfp*a}] for each ``a`` in ``_a_abs_list``.

        For ``a == 0`` the pair is ``(1, 0)`` (zonal seed).
        """
        Wr, Wi = jnp.ones_like(x), jnp.zeros_like(x)
        for _ in range(self.nfp):
            Wr, Wi = Wr * x - Wi * y, Wr * y + Wi * x

        out = []
        Pr, Pi = jnp.ones_like(x), jnp.zeros_like(x)
        for a_abs in self._a_abs_list:
            if a_abs > 0:
                Pr, Pi = Pr * Wr - Pi * Wi, Pr * Wi + Pi * Wr
            out.append((Pr, Pi))
        return out

    def ravel(self, weights: Weights) -> jnp.ndarray:
        """Pack a weight dict into a length-``num_modes`` dofs vector."""
        unknown = set(weights) - set(self._index)
        if unknown:
            raise KeyError(f"weight keys not in mode set: {sorted(unknown)}")
        return jnp.stack(
            [jnp.asarray(weights[k]) if k in weights else jnp.asarray(0.0)
             for k in self.keys]
        )

    def unravel(self, dofs) -> Dict[Key, jnp.ndarray]:
        """Unpack a length-``num_modes`` dofs vector into a dense weight dict."""
        dofs = jnp.asarray(dofs).reshape((self.num_modes,))
        return {k: dofs[i] for i, k in enumerate(self.keys)}

    def run_all(self, gamma) -> Dict[Key, jnp.ndarray]:
        """Evaluate every mode. ``gamma`` shape ``(..., 3)``."""
        return self._run_all_jit(gamma)

    @abc.abstractmethod
    def _run_all_impl(self, gamma) -> Dict[Key, jnp.ndarray]:
        ...

    @abc.abstractmethod
    def _phi_dofs_single_impl(self, p, dofs):
        """Potential at one point ``p: (3,)`` with dofs ``(K,)``."""
        ...

    def _grad_phi_dofs_single_impl(self, p, dofs):
        """Spatial gradient ``∇Φ`` at one point; returns ``(3,)``."""
        return jax.jacfwd(lambda q: self._phi_dofs_single_impl(q, dofs))(p)

    def _apply(self, fn, gamma, dofs, tail=()):
        """Batch/single dispatch shared by weighted potential and gradient."""
        gamma = jnp.asarray(gamma)
        w = jnp.asarray(dofs).reshape((self.num_modes,))
        if gamma.ndim == 1:
            return fn(gamma, w)
        shape = gamma.shape[:-1]
        flat = gamma.reshape((-1, 3))
        out = jax.vmap(fn, in_axes=(0, None))(flat, w)
        return out.reshape(shape + tail)

    def run_weighted_dofs(self, gamma, dofs) -> jnp.ndarray:
        """Weighted potential. ``gamma[..., 3]``, ``dofs: (num_modes,)``."""
        return self._apply(self._phi_dofs_single, gamma, dofs)

    def grad_run_weighted_dofs(self, gamma, dofs) -> jnp.ndarray:
        """Spatial gradient of the potential; shape ``(..., 3)``."""
        return self._apply(self._grad_phi_dofs_single, gamma, dofs, tail=(3,))


# ---------------------------------------------------------------------------
# recursion backend
# ---------------------------------------------------------------------------

class HarmonicRecursion(Harmonic):
    """Cartesian recursion with closed-form diagonal seeds.

    Single-point ``_phi_dofs_single`` / ``_grad_phi_dofs_single`` are the
    transport hot path. The gradient is ``jacfwd`` of the potential.

    Parameters
    ----------
    irregular
        If ``True`` (default), return irregular solid harmonics
        ``I_l^m = R_l^m / r^{2l+1}``. Singular at the origin; decays at
        infinity (better for bounded surface transport than regulars).
    """

    def __init__(
        self,
        nfp: int,
        a_max: int,
        l_max: int,
        stellsym: Optional[bool] = None,
        irregular: bool = True,
    ):
        super().__init__(nfp, a_max, l_max, stellsym=stellsym)
        self.irregular = bool(irregular)
        self._two_l_plus_one = jnp.asarray(
            [2 * l + 1 for l, _ in self.keys], dtype=jnp.int32
        )

        alpha: Dict[Tuple[int, int], float] = {}
        beta: Dict[Tuple[int, int], float] = {}
        for m in self._ms:
            for l in range(m + 1, self.l_max + 1):
                den = np.sqrt(l * l - m * m)
                alpha[(l, m)] = float((2 * l - 1) / den)
                if l - 1 > m:
                    beta[(l, m)] = float(np.sqrt((l - 1) ** 2 - m * m) / den)
        self._alpha = alpha
        self._beta = beta

    def _R_flat(self, x, y, z):
        """Mode values with shape ``(..., num_modes)``, columns as ``self.keys``.

        Regular recursion, then optional irregular post-scale ``/ r^{2l+1}``.
        """
        r2 = x * x + y * y + z * z
        R: dict = {}
        for a_abs, m, (Pr, Pi) in zip(
            self._a_abs_list, self._ms, self._angular_pairs(x, y)
        ):
            c = self._c_m[m]
            for s in self._signs(a_abs):
                mm = s * m
                prev1 = c * (Pr if s > 0 else Pi)
                prev2 = jnp.zeros_like(x)
                R[(m, mm)] = prev1
                for l in range(m + 1, self.l_max + 1):
                    t = self._alpha[(l, m)] * z * prev1
                    if (l, m) in self._beta:
                        t = t - self._beta[(l, m)] * r2 * prev2
                    prev2, prev1 = prev1, t
                    R[(l, mm)] = prev1

        cols = []
        for l, a in self.keys:
            m = self.nfp * abs(a)
            mm = m if a > 0 else -m
            cols.append(R[(l, mm)])
        vals = jnp.stack(cols, axis=-1)
        if self.irregular:
            r = jnp.sqrt(r2)
            vals = vals / (r[..., None] ** self._two_l_plus_one)
        return vals

    def _run_all_impl(self, gamma):
        shape, x, y, z = self._unpack_gamma(gamma)
        vals = self._R_flat(x, y, z)
        return {
            key: vals[:, j].reshape(shape) for j, key in enumerate(self.keys)
        }

    def _phi_dofs_single_impl(self, p, w):
        vals = self._R_flat(p[0:1], p[1:2], p[2:3])[0]
        return jnp.dot(vals, w)


# ---------------------------------------------------------------------------
# GEMM backend
# ---------------------------------------------------------------------------

class HarmonicGEMM(Harmonic):
    """Factored monomial evaluation via one GEMM.

    Prefer ``HarmonicRecursion`` for single-point flow / nested surface AD.
    Spatial gradients use ``jacfwd``.
    """

    def __init__(
        self,
        nfp: int,
        a_max: int,
        l_max: int,
        stellsym: Optional[bool] = None,
    ):
        super().__init__(nfp, a_max, l_max, stellsym=stellsym)

        Q = _q_poly_recursion(self.l_max, self._ms)
        monos = sorted({
            e for (l, a) in self.keys for e in Q[(l, self.nfp * abs(a))]
        })
        M = len(monos)
        mono_idx = {e: i for i, e in enumerate(monos)}

        K = len(self.keys)
        C = np.zeros((M, K), dtype=float)
        # Angular layout: a=0 always has one real slot; for a>0, cosine
        # only if stellsym else cosine+sine.
        ang_slots: dict = {}
        slot = 0
        for a in self._a_abs_list:
            if a == 0:
                ang_slots[0] = slot
                slot += 1
            elif self.stellsym:
                ang_slots[a] = slot
                slot += 1
            else:
                ang_slots[(a, False)] = slot
                ang_slots[(a, True)] = slot + 1
                slot += 2

        ang_index = np.zeros(K, dtype=np.int32)
        for j, (l, a) in enumerate(self.keys):
            m = self.nfp * abs(a)
            c = self._c_m[m]
            for e, coef in Q[(l, m)].items():
                C[mono_idx[e], j] = c * coef
            if a == 0:
                ang_index[j] = ang_slots[0]
            elif self.stellsym:
                ang_index[j] = ang_slots[abs(a)]
            else:
                ang_index[j] = ang_slots[(abs(a), a < 0)]

        self._C = jnp.asarray(C)
        self._ang_index = jnp.asarray(ang_index)
        self._max_p = max(p for p, q in monos)
        self._max_q = max(q for p, q in monos)
        self._mono_p = jnp.asarray([p for p, q in monos], dtype=jnp.int32)
        self._mono_q = jnp.asarray([q for p, q in monos], dtype=jnp.int32)

    def _angular_stack(self, x, y):
        cols = []
        for a_abs, (Pr, Pi) in zip(self._a_abs_list, self._angular_pairs(x, y)):
            cols.append(Pr)
            if a_abs > 0 and not self.stellsym:
                cols.append(Pi)
        return jnp.stack(cols, axis=-1)

    def _basis(self, x, y, z):
        r2 = x * x + y * y + z * z
        zp = [jnp.ones_like(z)]
        for _ in range(int(self._max_p)):
            zp.append(zp[-1] * z)
        zp = jnp.stack(zp, axis=-1)
        rq = [jnp.ones_like(r2)]
        for _ in range(int(self._max_q)):
            rq.append(rq[-1] * r2)
        rq = jnp.stack(rq, axis=-1)
        return zp[:, self._mono_p] * rq[:, self._mono_q]

    def _run_all_impl(self, gamma):
        shape, x, y, z = self._unpack_gamma(gamma)
        B = self._basis(x, y, z)
        Q = B @ self._C
        A = self._angular_stack(x, y)
        vals = Q * A[:, self._ang_index]
        return {
            key: vals[:, j].reshape(shape) for j, key in enumerate(self.keys)
        }

    def _phi_dofs_single_impl(self, p, w):
        x, y, z = p[0:1], p[1:2], p[2:3]
        B = self._basis(x, y, z)           # (1, M)
        Q = (B @ self._C)[0]               # (K,)
        A = self._angular_stack(x, y)[0]
        vals = Q * A[self._ang_index]
        return jnp.dot(vals, w)
