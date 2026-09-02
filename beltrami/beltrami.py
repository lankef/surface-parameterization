"""Morse (2005) cylindrical Beltrami eigenvalues, curl B = λ B.

Reference
---------
Edward C. Morse, *Eigenfunctions of the curl in cylindrical geometry*,
J. Math. Phys. **46**, 113511 (2005), DOI 10.1063/1.2118447.
Local copy: ``Morse - 2005 - Eigenfunctions of the curl in cylindrical geometry.pdf``.

Geometry and series (Sec. IV)
-----------------------------
Unit-length cylinder, ``Z ∈ [-1/2, 1/2]``, radius ``a``, azimuthal mode
``m ≥ 1``, truncation ``N = max_order``. Paper index ``n = 1…N`` is stored
at array index ``i = n - 1``.

The Chandrasekhar–Kendall generating function (anharmonic coefficient fixed
at 1; overall amplitude remains arbitrary) is

    χ(R, Z) = (R/a)^m exp(i λ Z)
        + Σ_n b_n [J_m(k_o[n] R) / J_m(k_o[n] a)] cos((2n−1) π Z)
        + i Σ_n a_n [J_m(k_e[n] R) / J_m(k_e[n] a)] sin(2 π n Z)

with radial wavenumbers

    k_o[n]^2 = λ^2 − (2n−1)^2 π^2 ,   k_e[n]^2 = λ^2 − 4 n^2 π^2 .

Negative ``k^2`` uses the modified-Bessel ``I_m`` branch (``J_m(iκ) = i^m I_m(κ)``).

Odd / even naming (paper warning, Sec. IV)
------------------------------------------
Morse labels “odd” / “even” by the *integer* in the trigonometric factor,
not by Z-parity of χ:

* odd-integer cosines ``cos((2n−1)π Z)`` are *even* functions of Z (``b_n``);
* even-integer sines ``sin(2n π Z)`` are *odd* functions of Z (``a_n``).

Code names follow the same convention: ``*_odd`` / ``kz_odd`` / ``bn`` are
the odd-integer cosine series; ``*_even`` / ``kz_even`` / ``an`` are the
even-integer sine series.

Boundary matching and eigenvalue criterion
------------------------------------------
Radial logarithmic derivatives ``Q_p`` and the sine-to-cosine remapping
``SC(n, n′)`` (Sec. IV) build the block matching system
``M · (a, b) = (u_even, u_odd)``. Morse then forms the inverse-component
matrices ``z1``, ``z2`` and solves the rank-``N`` systems of Eq. (4.1):

    Z_even · a = u1′ ,   Z_odd · b = u2′ .

This implementation never forms ``z1`` / ``z2``. It applies the algebraically
equivalent Schur reduction that eliminates ``a`` directly:

    (I − M_even M_odd) b = u_odd − M_even u_even ,
    a = u_even − M_odd b ,

where code identifiers ``u_cos ≡ u_odd``, ``u_sin ≡ u_even``,
``m_even ≡ (a/(mλ)) SC diag(Q_even)``, and
``m_odd ≡ (a/(mλ)) SCᵀ diag(Q_odd)``.

Eigenvalues are the zeros of Morse Eq. (4.2):

    ε(λ) = Σ_n b_n (−1)^n / (2n−1) − π sin(λ/2) / λ .

Notation map (paper → code)
---------------------------
=========  =====================  ========================================
Paper      Code                   Shape / notes
=========  =====================  ========================================
λ          lam                    scalar or (K,) trial eigenvalues
a          radius                 cylinder radius (length fixed at 1)
m          m                      azimuthal mode, m ≥ 1
N          max_order              retained terms in both series
a_n        an                     (K, N) even-integer sine coefficients
b_n        bn                     (K, N) odd-integer cosine coefficients
k_Z even   kz_even                (N,) = 2 n π
k_Z odd    kz_odd                 (N,) = (2n−1) π
k_e², k_o² kr2_even, kr2_odd      (K, N) signed radial wavenumber squared
Q_p        q_even / q_odd         (K, N) radial log-derivative at R = a
SC(n,n′)   sc                     (N, N); sc[i,j] = SC(n=j+1, n′=i+1)
u_n^odd    u_cos                  (K, N) anharmonic projection on cosines
u_n^even   u_sin                  (K, N) anharmonic projection on sines
ε(λ)       eps / residual         (K,) Morse Eq. (4.2)
=========  =====================  ========================================

Array axes used throughout: ``K`` = batch of trial λ, ``N`` = Fourier
terms, ``...`` = arbitrary broadcastable field-point batch. NumPy
broadcasting provides the λ-batch map; JAX broadcasting provides the
field-point map. This module does **not** call ``jax.vmap``,
``jax.lax.scan``, or chunk λ batches.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import ive, jn_zeros, jv, jvp

__all__ = [
    "CONVENTION",
    "VERSION",
    "IntervalStatus",
    "BeltramiSpectrum",
    "evaluate_kernel",
    "extrapolate_lam",
    "find_beltrami_lam",
    "find_beltrami_lam_axisym",
    "generate_beltrami_callable",
    "generate_beltrami_weighted_callable",
]

VERSION = "morse-2005-v1"

CONVENTION = (
    "chi(R,Z) = (R/a)^m exp(i*lam*Z) "
    "+ sum_n b_n Jm(k_o[n] R)/Jm(k_o[n] a) cos((2n-1)*pi*Z) "
    "+ i sum_n a_n Jm(k_e[n] R)/Jm(k_e[n] a) sin(2*pi*n*Z); "
    "k_o[n]^2 = lam^2-(2n-1)^2 pi^2, k_e[n]^2 = lam^2-4 n^2 pi^2; "
    "negative k^2 uses the I_m branch; anharmonic coefficient is 1; "
    "array index i is paper n=i+1; cylinder length 1, Z in [-1/2, 1/2]."
)

# Branch threshold for signed k^2 (oscillatory J vs evanescent I vs power).
_KR2_EPS = 1e-12
# Removable-resonance threshold for sin(δ/2)/δ → 1/2 as δ → 0.
_SINC_EPS = 1e-12
# Floor on R (and Bessel arguments) to keep 1/R and recurrence formulas finite.
_R_FLOOR = 1e-12
# JAX ``bessel_jn`` returns NaN for ``|z| ≲ 2e-5``; floor the argument and
# replace with the small-z series below this threshold.
_BESSEL_JN_FLOOR = 2e-5


class IntervalStatus(IntEnum):
    """Per-interval outcome of the λ-grid search in ``find_beltrami_lam``.

    Each consecutive pair of the λ grid is one independent bisection
    bracket. The status array has length ``n_lam`` (one entry per bracket)
    and records why that bracket was kept or dropped. Morse Sec. V notes
    that genuine eigenvalues are the *smooth* zero crossings of ε(λ), while
    λ = pπ produces a trivial B ≡ 0 field (rejected as ``SPURIOUS_PI``).
    Normalization poles of J_m(k a) (Sec. IV) appear as ``POLE_CROSSING``.
    """

    ROOT = 0
    NO_SIGN_CHANGE = 1
    NONFINITE = 2
    SPURIOUS_PI = 3
    POLE_CROSSING = 4
    MAX_ITER = 5
    RESIDUAL_TOO_LARGE = 6
    DUPLICATE = 7


@dataclass(frozen=True)
class BeltramiSpectrum:
    """Accepted Morse eigenmodes plus the per-interval search status.

    Fields required by ``generate_beltrami_callable`` (Sec. III CK field
    from the Sec. IV series) are ``m``, ``radius``, ``lam``, ``an``, ``bn``,
    ``kr2_even``, ``kr2_odd``, ``kz_even``, ``kz_odd``, and ``anharmonic``.
    The remaining fields are diagnostics from the root search.

    Attributes
    ----------
    m, radius, max_order
        Paper ``m``, ``a``, and truncation ``N``.
    lam : (n_modes,)
        Accepted eigenvalues λ (roots of Eq. (4.2), or analytic Sec. II
        values when ``m = 0``).
    residual : (n_modes,)
        ε(λ) at each accepted root (zeros for analytic Sec. II modes).
    an, bn : (n_modes, N)
        Paper coefficients ``a_n`` (even-integer sines) and ``b_n``
        (odd-integer cosines). For ``m = 0`` these are one-hot encodings
        of the single Sec. II trigonometric term.
    kr2_even, kr2_odd : (n_modes, N)
        Signed ``k_e[n]^2`` and ``k_o[n]^2``.
    kz_even, kz_odd : (N,)
        Axial wave numbers ``2 n π`` and ``(2n−1) π``.
    min_bessel_denom : (n_modes,)
        Smallest |J_m(k a)| over real-argument terms; near-zero warns of
        the Sec. IV normalization poles. For ``m = 0`` this is
        ``|J_0(k_p a)|`` of the selected radial root.
    status : (n_lam,)
        ``IntervalStatus`` per search bracket (not per mode). Analytic
        Sec. II spectra use a length-``n_modes`` array of ``ROOT``.
    anharmonic
        Coefficient on the Sec. IV seed ``(R/a)^m exp(i λ Z)``. ``1.0``
        for Sec. IV modes (``m >= 1``); ``0.0`` for Sec. II axisymmetric
        modes, which have no anharmonic seed.
    """

    m: int
    radius: float
    max_order: int
    lam: NDArray[np.floating]
    residual: NDArray[np.floating]
    an: NDArray[np.floating]
    bn: NDArray[np.floating]
    kr2_even: NDArray[np.floating]
    kr2_odd: NDArray[np.floating]
    kz_even: NDArray[np.floating]
    kz_odd: NDArray[np.floating]
    min_bessel_denom: NDArray[np.floating]
    status: NDArray[np.int_]
    anharmonic: float = 1.0

    @property
    def n_modes(self) -> int:
        """Number of accepted eigenvalues (length of ``lam``)."""
        return int(self.lam.shape[0])


# ---------------------------------------------------------------------------
# Kernel (Sec. IV matching system and Eq. (4.2))
# ---------------------------------------------------------------------------


def _precompute(max_order: int) -> dict[str, NDArray[Any]]:
    """Build λ-independent Sec. IV quantities for truncation ``N = max_order``.

    Returns a dict with:

    * ``n`` — paper indices ``1…N``, shape ``(N,)``;
    * ``kz_even`` — ``2 n π``, shape ``(N,)``;
    * ``kz_odd`` — ``(2n−1) π``, shape ``(N,)``;
    * ``sc`` — Morse sine-to-cosine operator
      ``SC(n, n′) = 8 (−1)^{n′+n} (1−2n′) n / ((2n′−1)^2 − 4 n^2)``,
      stored so that ``sc[i, j] = SC(n=j+1, n′=i+1)`` with shape ``(N, N)``.
      Row index tracks the odd-integer (cosine / ``b``) slot; column index
      tracks the even-integer (sine / ``a``) slot.
    """
    n = np.arange(1, max_order + 1, dtype=np.intp)
    # Meshgrid over paper indices: nn ↔ n′ (odd), npp ↔ n (even).
    nn, npp = np.meshgrid(n, n, indexing="ij")
    sc = (
        8.0
        * ((-1.0) ** (nn + npp))
        * (1.0 - 2.0 * nn)
        * npp
        / ((2.0 * nn - 1.0) ** 2 - 4.0 * npp**2)
    )
    return {
        "n": n,
        "sc": np.ascontiguousarray(sc),
        "kz_even": 2.0 * np.pi * n.astype(float),
        "kz_odd": (2.0 * n.astype(float) - 1.0) * np.pi,
    }


def _forcing(
    lam: NDArray[np.floating], n: NDArray[np.integer]
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Anharmonic forcing vectors ``u_odd`` and ``u_even`` (Sec. IV).

    Paper formulas (with removable poles at ``λ = p π``):

        u_n'^odd  = 4 (−1)^{n′} (1−2n′) π cos(λ/2) / ((1−2n′)^2 π^2 − λ^2)
        u_n^even  = 8 (−1)^n n π sin(λ/2) / ((2 n π)^2 − λ^2)

    These are algebraically rewritten around ``δ = λ − p π`` so that
    ``sin(δ/2)/δ → 1/2`` as ``δ → 0``, giving the finite limits
    ``u → −1`` on the resonant entry (tested in the suite).

    Parameters
    ----------
    lam : (K,)
        Trial eigenvalues.
    n : (N,)
        Paper indices ``1…N``.

    Returns
    -------
    u_cos, u_sin : (K, N)
        ``u_cos ≡ u_odd`` (projects onto odd-integer cosines / ``b`` side);
        ``u_sin ≡ u_even`` (projects onto even-integer sines / ``a`` side).

    Shape ledger
    ------------
    ``lam[:, None]`` broadcasts λ against every Fourier index;
    ``n[None, :]`` broadcasts the index against every trial λ.
    """
    # lam_col: (K, 1); n_row: (1, N) → all results (K, N).
    lam_col = np.asarray(lam, dtype=float)[:, None]
    n_row = n.astype(float)[None, :]
    p_odd = 2.0 * n_row - 1.0   # paper (2n′ − 1)
    p_even = 2.0 * n_row        # paper 2n
    delta_odd = lam_col - p_odd * np.pi
    delta_even = lam_col - p_even * np.pi
    # Removable sinc: sin(δ/2)/δ → 1/2 as δ → 0 (λ = pπ resonance).
    sinc_odd = np.divide(
        np.sin(0.5 * delta_odd),
        delta_odd,
        out=np.full_like(delta_odd, 0.5),
        where=np.abs(delta_odd) >= _SINC_EPS,
    )
    sinc_even = np.divide(
        np.sin(0.5 * delta_even),
        delta_even,
        out=np.full_like(delta_even, 0.5),
        where=np.abs(delta_even) >= _SINC_EPS,
    )
    u_cos = -4.0 * np.pi * p_odd * sinc_odd / (lam_col + p_odd * np.pi)
    u_sin = -8.0 * np.pi * n_row * sinc_even / (lam_col + p_even * np.pi)
    return u_cos, u_sin


def _q_and_jm(
    m: int, radius: float, kr2: NDArray[np.floating]
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Morse radial log-derivative ``Q_p`` (Sec. IV) and ``J_m(k a)``.

    Paper definition (``p`` even or odd axial integer):

        Q_p = √(λ² − p²π²)
              · (J_{m−1}(a√…) − J_{m+1}(a√…))
              / (2 J_m(a√…))

    which equals ``k J_m′(k a) / J_m(k a)`` on the oscillatory branch.
    Three branches:

    * ``kr2 > 0`` — oscillatory ``J_m`` / ``J_m′``;
    * ``kr2 < 0`` — evanescent ``I_m`` via scaled ``ive``
      (``I_m′(x)/I_m(x) = ½(I_{m−1}+I_{m+1})/I_m``);
    * ``kr2 ≈ 0`` — power-law limit ``Q = m / a``.

    Parameters
    ----------
    kr2 : (K, N)
        Signed radial wavenumber squared.

    Returns
    -------
    q, jm, signs : (K, N)
        ``Q_p``; ``J_m(k a)`` on the J branch else 0; ``sign(J_m)`` on the
        J branch else 0 (used to detect Sec. IV normalization poles across
        a λ bracket).
    """
    q = np.empty_like(kr2, dtype=float)
    jm = np.zeros_like(kr2, dtype=float)
    # Boolean masks over the flat (K, N) grid; each branch filled in place.
    pos = kr2 > _KR2_EPS
    neg = kr2 < -_KR2_EPS
    zer = ~(pos | neg)
    if np.any(pos):
        # Gather oscillatory entries, evaluate, scatter back via boolean index.
        k = np.sqrt(kr2[pos])
        xa = k * radius
        jval = jv(m, xa)
        jm[pos] = jval
        with np.errstate(divide="ignore", invalid="ignore"):
            q[pos] = k * jvp(m, xa, 1) / jval
    if np.any(neg):
        kappa = np.sqrt(-kr2[neg])
        xa = kappa * radius
        with np.errstate(divide="ignore", invalid="ignore"):
            q[neg] = (
                kappa
                * 0.5
                * (ive(m - 1, xa) + ive(m + 1, xa))
                / ive(m, xa)
            )
    q[zer] = float(m) / float(radius)
    signs = np.sign(jm)
    signs[~pos] = 0.0
    return q, jm, signs


@dataclass
class _KernelOut:
    """Packed output of one batched ε(λ) evaluation.

    All leading axes have length ``K = len(lam)``. Coefficient fields are
    ``None`` unless ``full=True`` was requested.
    """

    eps: NDArray[np.floating]
    kr2_even: NDArray[np.floating]
    kr2_odd: NDArray[np.floating]
    jm_sign_even: NDArray[np.floating]
    jm_sign_odd: NDArray[np.floating]
    bn: NDArray[np.floating] | None = None
    an: NDArray[np.floating] | None = None
    min_bessel_denom: NDArray[np.floating] | None = None


def _eps_batch(
    lam: NDArray[np.floating],
    m: int,
    radius: float,
    pre: dict[str, NDArray[Any]],
    *,
    full: bool = False,
) -> _KernelOut:
    """Evaluate Morse Eq. (4.2) for a batch of trial λ.

    Concrete steps (one NumPy broadcast over the λ batch — no chunking,
    no ``vmap``):

    1. Form signed radial wavenumbers ``k_e²``, ``k_o²``.
    2. Evaluate ``Q_even``, ``Q_odd`` and the anharmonic forcings.
    3. Build ``M_even``, ``M_odd`` and the Schur-reduced system for ``b``.
    4. Solve for ``b_n``; form ``ε(λ)`` from Eq. (4.2).
    5. If ``full``, recover ``a_n = u_even − M_odd b`` and the Bessel-pole
       diagnostic.

    Shape ledger
    ------------
    ================  ==========  ===========================================
    Expression        Shape       Role
    ================  ==========  ===========================================
    lam               (K,)        trial eigenvalues
    kz_*[None, :]     (1, N)      axial wave numbers vs every λ
    lam[:, None]      (K, 1)      λ vs every Fourier term
    kr2_*             (K, N)      signed k²
    q_*, u_*          (K, N)      Q_p and forcings
    c[:, None, None]  (K, 1, 1)   a/(m λ) vs every matrix entry
    sc[None, :, :]    (1, N, N)   SC vs every λ
    q_even[:, None,:] (K, 1, N)   diag(Q_even) via broadcast multiply
    m_even, m_odd     (K, N, N)   matching matrices
    system, rhs       (K,N,N)/(K,N)  Schur system for b
    bn_all, eps       (K, N)/(K,) coefficients and Eq. (4.2)
    ================  ==========  ===========================================
    """
    lam = np.asarray(lam, dtype=float).reshape(-1)
    n = pre["n"]
    sc = pre["sc"]
    batch = lam.shape[0]
    order = n.shape[0]

    # Step 1: k_e[n]^2 = λ^2 − (2nπ)^2 , k_o[n]^2 = λ^2 − ((2n−1)π)^2.
    kr2_even = lam[:, None] ** 2 - pre["kz_even"][None, :] ** 2
    kr2_odd = lam[:, None] ** 2 - pre["kz_odd"][None, :] ** 2
    # Step 2: Q_p (Sec. IV) and anharmonic forcings u_odd, u_even.
    q_even, jm_even, sign_even = _q_and_jm(m, radius, kr2_even)
    q_odd, jm_odd, sign_odd = _q_and_jm(m, radius, kr2_odd)
    u_cos, u_sin = _forcing(lam, n)  # u_cos ≡ u_odd, u_sin ≡ u_even

    # Step 3: M_even[b] = c[b] · SC · diag(Q_even[b])
    #         M_odd[b]  = c[b] · SCᵀ · diag(Q_odd[b])
    # with c = a/(m λ). Broadcast multiply builds the (K, N, N) stacks.
    c = radius / (m * lam)
    m_even = c[:, None, None] * sc[None, :, :] * q_even[:, None, :]
    m_odd = c[:, None, None] * sc.T[None, :, :] * q_odd[:, None, :]
    eye = np.eye(order)
    # Schur reduction of the block system (avoids Morse's z1/z2 inverses):
    #   (I − M_even M_odd) b = u_odd − M_even u_even .
    system = eye[None, :, :] - np.matmul(m_even, m_odd)
    # u_sin[:, :, None] → (K, N, 1) so matmul yields (K, N, 1); [..., 0]
    # drops the trailing singleton back to (K, N).
    rhs = u_cos - np.matmul(m_even, u_sin[:, :, None])[..., 0]

    # Step 4: solve for b_n and evaluate Eq. (4.2).
    eps = np.full(batch, np.nan)
    bn: NDArray[np.floating] | None = None
    an: NDArray[np.floating] | None = None
    finite = np.isfinite(system).all(axis=(1, 2)) & np.isfinite(rhs).all(axis=1)
    bn_all = np.full((batch, order), np.nan)
    if np.any(finite):
        try:
            # Batched LAPACK over the finite λ-slice; [..., 0] drops the
            # RHS column dimension introduced by solve.
            bn_all[finite] = np.linalg.solve(
                system[finite], rhs[finite][..., None]
            )[..., 0]
        except np.linalg.LinAlgError:
            # Fallback: the whole batch failed as one LAPACK call, so walk
            # finite rows one-by-one and leave singular rows as NaN.
            for i in np.flatnonzero(finite):
                try:
                    bn_all[i] = np.linalg.solve(system[i], rhs[i])
                except np.linalg.LinAlgError:
                    bn_all[i] = np.nan
        # Eq. (4.2): ε = Σ_n b_n (−1)^n/(2n−1) − π sin(λ/2)/λ .
        sign_n = (-1.0) ** n
        crit_sum = np.sum(bn_all * sign_n[None, :] / (2.0 * n[None, :] - 1.0), axis=1)
        anharmonic = np.pi * np.sin(0.5 * lam) / lam
        eps = crit_sum - anharmonic

    if full:
        # Recover a = u_even − M_odd b (second block equation).
        an = u_sin - np.matmul(m_odd, bn_all[:, :, None])[..., 0]
        bn = bn_all

    min_bessel = None
    if full:
        # Diagnostic only: smallest |J_m(k a)| over oscillatory terms.
        # Inf on the I / zero branches so they do not win the min.
        abs_even = np.abs(jm_even)
        abs_even[kr2_even <= _KR2_EPS] = np.inf
        abs_odd = np.abs(jm_odd)
        abs_odd[kr2_odd <= _KR2_EPS] = np.inf
        min_bessel = np.minimum(np.min(abs_even, axis=-1), np.min(abs_odd, axis=-1))

    return _KernelOut(
        eps=eps,
        kr2_even=kr2_even,
        kr2_odd=kr2_odd,
        jm_sign_even=sign_even,
        jm_sign_odd=sign_odd,
        bn=bn,
        an=an,
        min_bessel_denom=min_bessel,
    )


def evaluate_kernel(
    m: int,
    lam: Any,
    max_order: int,
    radius: float = 1.0,
) -> dict[str, NDArray[Any]]:
    """Evaluate ``a_n``, ``b_n``, ε(λ), and signed ``k²`` (no root search).

    Thin public wrapper around ``_eps_batch(..., full=True)`` for inspecting
    the Sec. IV kernel at chosen λ values — useful for decay-law checks
    (Sec. V: ``a_n, b_n ∼ n⁻⁴`` on a root; ``a_n ∼ n⁻²`` off a root).

    Parameters
    ----------
    m, max_order, radius
        Paper ``m``, truncation ``N``, and cylinder radius ``a``.
    lam
        Scalar or array of trial eigenvalues.

    Returns
    -------
    dict
        Keys ``lam``, ``eps``, ``an``, ``bn``, ``kr2_even``, ``kr2_odd``,
        ``min_bessel_denom``. Scalars are returned when ``lam`` is scalar.
    """
    if not isinstance(m, (int, np.integer)) or int(m) < 1:
        raise ValueError(f"m must be an integer >= 1, got {m!r}")
    if not isinstance(max_order, (int, np.integer)) or int(max_order) < 2:
        raise ValueError(f"max_order must be an integer >= 2, got {max_order!r}")
    if not (radius > 0.0):
        raise ValueError(f"radius must be > 0, got {radius!r}")
    lam_arr = np.asarray(lam, dtype=float)
    scalar = lam_arr.ndim == 0
    pre = _precompute(int(max_order))
    out = _eps_batch(lam_arr.reshape(-1), int(m), float(radius), pre, full=True)
    result = {
        "lam": np.asarray(lam_arr.reshape(-1), dtype=float).copy(),
        "eps": out.eps,
        "an": out.an,
        "bn": out.bn,
        "kr2_even": out.kr2_even,
        "kr2_odd": out.kr2_odd,
        "min_bessel_denom": out.min_bessel_denom,
    }
    if scalar:
        return {key: value[0] for key, value in result.items()}
    return result


def _tail_exponent(values: NDArray[np.floating], n: NDArray[np.integer]) -> float:
    """Log-log slope of ``|a_n|`` or ``|b_n|`` over ``n ∈ [N/4, 0.9 N]``.

    Sec. V reports ``∼ n⁻⁴`` on an eigenvalue and ``a_n ∼ n⁻²`` off one.
    The uppermost ∼10 % of coefficients are truncation-contaminated and
    are excluded from the fit window. Returns NaN if fewer than two
    positive finite samples remain.
    """
    order = int(n.size)
    lo = max(int(np.ceil(order / 4.0)), 1)
    hi = max(int(np.floor(0.9 * order)), lo)
    # n is 1-indexed; convert the inclusive paper window to a 0-based slice.
    start = lo - 1
    sl = slice(start, hi)
    mag = np.abs(np.asarray(values, dtype=float)[sl])
    nn = n[sl].astype(float)
    ok = np.isfinite(mag) & (mag > 0.0) & (nn > 0.0)
    if int(ok.sum()) < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(nn[ok]), np.log(mag[ok]), 1)
    return float(slope)


def extrapolate_lam(orders: Any, lams: Any) -> float:
    """Richardson extrapolation of ``λ(N)`` in powers of ``1/N²``.

    Sec. V observes that the truncation error in λ is ``O(1/N⁴)`` for
    Eq. (4.2) itself, while self-convergence of the computed root is
    empirically ``∼ 1/N²``. This helper (not a numbered paper equation)
    fits

    * two points → ``λ = λ_∞ + C / N²``;
    * three or more → last three points with an extra ``1/N⁴`` term.

    Returns the intercept ``λ_∞``.
    """
    n_arr = np.asarray(orders, dtype=float).reshape(-1)
    lam_arr = np.asarray(lams, dtype=float).reshape(-1)
    if n_arr.size != lam_arr.size or n_arr.size < 2:
        raise ValueError("extrapolate_lam needs at least two matched (order, lam) pairs")
    if np.any(n_arr <= 0):
        raise ValueError("orders must be positive")
    if n_arr.size == 2:
        x = 1.0 / n_arr**2
        a = np.column_stack([np.ones(2), x])
        coef, *_ = np.linalg.lstsq(a, lam_arr, rcond=None)
        return float(coef[0])
    # Use only the three finest truncations for the 1/N² + 1/N⁴ fit.
    n_arr = n_arr[-3:]
    lam_arr = lam_arr[-3:]
    x = 1.0 / n_arr**2
    a = np.column_stack([np.ones(3), x, x**2])
    coef, *_ = np.linalg.lstsq(a, lam_arr, rcond=None)
    return float(coef[0])


# ---------------------------------------------------------------------------
# Search (Sec. V: locate roots of Eq. (4.2))
# ---------------------------------------------------------------------------


def find_beltrami_lam(
    m: int,
    min_lam: float,
    max_lam: float,
    n_lam: int,
    max_order: int,
    max_iter: int,
    tol: float,
    *,
    radius: float = 1.0,
    pi_guard: float = 1e-3,
    polish: bool = True,
) -> BeltramiSpectrum:
    """Locate Beltrami eigenvalues by batched bisection of Morse Eq. (4.2).

    Algorithm
    ---------
    1. Evaluate ε(λ) on a uniform grid of ``n_lam + 1`` endpoints.
    2. Classify each consecutive bracket: sign change, non-finite, or
       ``J_m(k a)`` pole crossing (sign of ``J_m`` flips inside the bracket).
    3. Batched bisection over all active brackets in lockstep (one
       ``_eps_batch`` mid-point call per sweep).
    4. Optional scalar ``brentq`` polish inside each converged bracket.
    5. Reject Sec. V spurious zeros at ``λ = p π``, oversize residuals,
       and near-duplicate roots.
    6. Re-evaluate accepted roots with ``full=True`` to recover ``a_n``,
       ``b_n``, and signed ``k²``.

    Parameters
    ----------
    m
        Azimuthal mode number, ``m >= 1``.
    min_lam, max_lam, n_lam
        Search range. Endpoints are ``linspace(min_lam, max_lam, n_lam+1)``.
        Each consecutive pair is one independent bisection interval.
    max_order
        Truncation ``N``. The first ``N`` terms of both series are kept.
    max_iter, tol
        Bisection budget and residual / relative-width tolerance.
    radius
        Cylinder radius ``a`` (length is fixed at 1).
    pi_guard
        Reject a candidate if ``min_p |λ − p π| / π < pi_guard``
        (Sec. V trivial-field zeros).
    polish
        Run ``brentq`` inside each converged bracket.

    Returns
    -------
    BeltramiSpectrum
        Accepted modes plus a per-interval ``status``. Intervals without a
        genuine root are recorded, not invented.
    """
    if not isinstance(m, (int, np.integer)) or int(m) < 1:
        raise ValueError(f"m must be an integer >= 1, got {m!r}")
    if not (min_lam > 0.0 and max_lam > min_lam):
        raise ValueError(
            f"need 0 < min_lam < max_lam, got min_lam={min_lam!r}, max_lam={max_lam!r}"
        )
    if not isinstance(n_lam, (int, np.integer)) or int(n_lam) < 1:
        raise ValueError(f"n_lam must be an integer >= 1, got {n_lam!r}")
    if not isinstance(max_order, (int, np.integer)) or int(max_order) < 2:
        raise ValueError(f"max_order must be an integer >= 2, got {max_order!r}")
    if not isinstance(max_iter, (int, np.integer)) or int(max_iter) < 1:
        raise ValueError(f"max_iter must be an integer >= 1, got {max_iter!r}")
    if not (tol > 0.0):
        raise ValueError(f"tol must be > 0, got {tol!r}")
    if not (radius > 0.0):
        raise ValueError(f"radius must be > 0, got {radius!r}")
    if not (pi_guard > 0.0):
        raise ValueError(f"pi_guard must be > 0, got {pi_guard!r}")

    m = int(m)
    n_lam = int(n_lam)
    max_order = int(max_order)
    max_iter = int(max_iter)
    radius = float(radius)
    tol = float(tol)
    pi_guard = float(pi_guard)
    pre = _precompute(max_order)

    # --- Step 1: ε and J_m signs on the uniform λ grid ---------------------
    grid = np.linspace(min_lam, max_lam, n_lam + 1)
    grid_out = _eps_batch(grid, m, radius, pre, full=False)
    eps_grid = grid_out.eps
    sign_even = grid_out.jm_sign_even
    sign_odd = grid_out.jm_sign_odd

    # Bracket endpoints: left/right are consecutive grid nodes.
    # grid[:-1] / grid[1:] → (n_lam,) paired edges of each search interval.
    left = grid[:-1].copy()
    right = grid[1:].copy()
    eps_left = eps_grid[:-1].copy()
    eps_right = eps_grid[1:].copy()
    status = np.full(n_lam, int(IntervalStatus.NO_SIGN_CHANGE), dtype=int)

    # --- Step 2: classify brackets -----------------------------------------
    finite = np.isfinite(eps_left) & np.isfinite(eps_right)
    status[~finite] = int(IntervalStatus.NONFINITE)
    sign_change = finite & (eps_left * eps_right < 0.0)
    # Pole crossing: sign(J_m) flips on any Fourier term between the two
    # endpoints (Sec. IV normalization pole of J_m(k a)).
    # sign_*[:-1] / sign_*[1:] pair endpoint rows; product < 0 on axis=1
    # means at least one term flipped inside that bracket.
    pole_even = finite & np.any(sign_even[:-1] * sign_even[1:] < 0.0, axis=1)
    pole_odd = finite & np.any(sign_odd[:-1] * sign_odd[1:] < 0.0, axis=1)
    pole = pole_even | pole_odd
    status[sign_change & pole] = int(IntervalStatus.POLE_CROSSING)

    # Gather active brackets into compact (n_active,) working arrays.
    # active_idx maps local working index → global interval index.
    active_idx = np.flatnonzero(sign_change)
    lo = left[active_idx].copy()
    hi = right[active_idx].copy()
    elo = eps_left[active_idx].copy()
    ehi = eps_right[active_idx].copy()
    still = np.ones(active_idx.size, dtype=bool)

    # --- Step 3: batcheded bisection over active brackets -------------------
    # Each sweep evaluates ε at every still-live midpoint in one
    # _eps_batch call (NumPy broadcast over the live λ subset).
    for _sweep in range(max_iter):
        # live: local indices into the compact active arrays that still need work.
        live = np.flatnonzero(still)
        if live.size == 0:
            break
        mid = 0.5 * (lo[live] + hi[live])
        emid = _eps_batch(mid, m, radius, pre, full=False).eps
        # Standard bisection update, vectorized over the live subset.
        same_as_lo = emid * elo[live] > 0.0
        lo[live] = np.where(same_as_lo, mid, lo[live])
        elo[live] = np.where(same_as_lo, emid, elo[live])
        hi[live] = np.where(~same_as_lo, mid, hi[live])
        ehi[live] = np.where(~same_as_lo, emid, ehi[live])
        width_ok = (hi[live] - lo[live]) <= tol * np.maximum(1.0, np.abs(mid))
        resid_ok = np.abs(emid) <= tol
        bad = ~np.isfinite(emid)
        still[live] = ~(width_ok | resid_ok | bad)

    # Scatter tightened brackets back into the full-length interval arrays.
    left[active_idx] = lo
    right[active_idx] = hi
    eps_left[active_idx] = elo
    eps_right[active_idx] = ehi
    unfinished = still
    status[active_idx[unfinished]] = int(IntervalStatus.MAX_ITER)

    # Candidates: active brackets that finished without hitting MAX_ITER.
    converged_mask = (status[active_idx] != int(IntervalStatus.MAX_ITER)) & ~unfinished
    cand_local = np.flatnonzero(converged_mask)
    cand_interval = active_idx[cand_local]
    cand_lo = lo[cand_local]
    cand_hi = hi[cand_local]
    cand_status = status[cand_interval].copy()

    cand_lam = 0.5 * (cand_lo + cand_hi)
    min_rtol = 4.0 * float(np.finfo(float).eps)
    if polish and cand_lam.size:
        # SciPy brentq is scalar-only, so polish one bracket at a time.
        def _scalar_eps(x: float) -> float:
            value = float(_eps_batch(np.array([x]), m, radius, pre, full=False).eps[0])
            if not np.isfinite(value):
                return np.copysign(1e30, x)
            return value

        for j in range(cand_lam.size):
            a_b, b_b = float(cand_lo[j]), float(cand_hi[j])
            fa, fb = _scalar_eps(a_b), _scalar_eps(b_b)
            if not (np.isfinite(fa) and np.isfinite(fb) and fa * fb < 0.0):
                continue
            try:
                cand_lam[j] = brentq(
                    _scalar_eps,
                    a_b,
                    b_b,
                    xtol=max(min_rtol * max(1.0, abs(a_b)), 0.0),
                    rtol=min_rtol,
                    maxiter=80,
                )
            except ValueError:
                pass

    if cand_lam.size:
        cand_eps = _eps_batch(cand_lam, m, radius, pre, full=False).eps
        # Prefer the lowest-|ε| sample among polished root and both ends.
        for j in range(cand_lam.size):
            samples = [
                (float(cand_lam[j]), float(cand_eps[j])),
                (float(cand_lo[j]), float(eps_left[int(cand_interval[j])])),
                (float(cand_hi[j]), float(eps_right[int(cand_interval[j])])),
            ]
            finite_samples = [(x, e) for x, e in samples if np.isfinite(e)]
            if finite_samples:
                best_x, best_e = min(finite_samples, key=lambda pair: abs(pair[1]))
                cand_lam[j] = best_x
                cand_eps[j] = best_e
    else:
        cand_eps = np.empty(0, dtype=float)

    # --- Step 5: accept / reject (Sec. V filters) --------------------------
    # |λ/π − round(λ/π)| is the normalized distance to the nearest pπ.
    pi_dist = np.abs(cand_lam / np.pi - np.round(cand_lam / np.pi))
    accepted: list[int] = []
    for j in range(cand_lam.size):
        interval_i = int(cand_interval[j])
        # Sec. V: ε crosses zero at λ = pπ because the anharmonic term
        # coincides with a basis function, forcing the trivial B ≡ 0 field.
        if float(pi_dist[j]) < pi_guard:
            status[interval_i] = int(IntervalStatus.SPURIOUS_PI)
            continue
        if not np.isfinite(cand_eps[j]) or abs(float(cand_eps[j])) > tol:
            if cand_status[j] == int(IntervalStatus.POLE_CROSSING):
                status[interval_i] = int(IntervalStatus.POLE_CROSSING)
            else:
                status[interval_i] = int(IntervalStatus.RESIDUAL_TOO_LARGE)
            continue
        accepted.append(j)

    if accepted:
        acc = np.asarray(accepted, dtype=int)
        # Sort by λ so adjacent-bracket duplicates become neighbors.
        order = np.argsort(cand_lam[acc])
        acc = acc[order]
        keep = np.ones(acc.size, dtype=bool)
        for p in range(1, acc.size):
            prev = acc[p - 1]
            cur = acc[p]
            sep = max(tol, 1e-9 * abs(float(cand_lam[cur])))
            if abs(float(cand_lam[cur] - cand_lam[prev])) < sep:
                # Keep the candidate with the smaller |ε|.
                if abs(float(cand_eps[cur])) < abs(float(cand_eps[prev])):
                    keep[p - 1] = False
                    status[int(cand_interval[prev])] = int(IntervalStatus.DUPLICATE)
                else:
                    keep[p] = False
                    status[int(cand_interval[cur])] = int(IntervalStatus.DUPLICATE)
        acc = acc[keep]
    else:
        acc = np.empty(0, dtype=int)

    # --- Step 6: recover full coefficients at accepted roots ---------------
    roots = cand_lam[acc] if acc.size else np.empty(0, dtype=float)
    recovered = _eps_batch(roots, m, radius, pre, full=True)
    for interval_i in cand_interval[acc]:
        status[int(interval_i)] = int(IntervalStatus.ROOT)

    return BeltramiSpectrum(
        m=m,
        radius=radius,
        max_order=max_order,
        lam=np.asarray(roots, dtype=float),
        residual=np.asarray(recovered.eps, dtype=float),
        an=np.asarray(recovered.an, dtype=float),
        bn=np.asarray(recovered.bn, dtype=float),
        kr2_even=np.asarray(recovered.kr2_even, dtype=float),
        kr2_odd=np.asarray(recovered.kr2_odd, dtype=float),
        kz_even=pre["kz_even"].copy(),
        kz_odd=pre["kz_odd"].copy(),
        min_bessel_denom=np.asarray(recovered.min_bessel_denom, dtype=float),
        status=status,
    )


def find_beltrami_lam_axisym(
    min_lam: float,
    max_lam: float,
    radius: float,
    max_order: int = 50,
) -> BeltramiSpectrum:
    """Enumerate Morse Sec. II axisymmetric (``m = 0``) Beltrami eigenvalues.

    Analytic closed form — no root search. Sec. II modes are

        χ = J_0(k_p R) T_q(Z) ,   k_p = j_{1,p} / a ,   λ² = k_p² + q² π² ,

    with ``J_0'(k_p a) = −J_1(k_p a) = 0`` (``p ≥ 1``; the trivial ``k = 0``
    root of ``J_1`` is excluded because it forces ``λ = q π`` and ``B ≡ 0``)
    and axial index ``q = 1 … 2 N`` so every retained Fourier slot of the
    Sec. IV series can host a mode.

    Each mode is encoded as a one-hot ``BeltramiSpectrum`` so
    ``generate_beltrami_callable`` can rebuild the field through the same
    path used for ``m ≥ 1``:

    * ``q`` odd (``q = 2n−1``): ``bn`` one-hot at index ``(q−1)//2``
    * ``q`` even (``q = 2n``): ``an`` one-hot at index ``q//2 − 1``
    * ``anharmonic = 0`` (Sec. II has no ``(R/a)^m exp(i λ Z)`` seed)

    Even-``q`` modes ride the ``an`` slot, which the series multiplies by an
    explicit factor of ``i``, so those fields emerge as ``i × (real field)``
    — an arbitrary global eigenmode phase, left as-is.

    Parameters
    ----------
    min_lam, max_lam
        Keep modes with ``min_lam <= λ <= max_lam``.
    radius
        Cylinder radius ``a`` (length is fixed at 1).
    max_order
        Truncation ``N``. Axial index runs ``q = 1 … 2 N``.

    Returns
    -------
    BeltramiSpectrum
        ``m = 0``, ``anharmonic = 0``, modes sorted by ascending ``λ``.
        ``residual`` is identically zero; ``status`` is ``ROOT`` per mode.
    """
    if not (min_lam > 0.0 and max_lam > min_lam):
        raise ValueError(
            f"need 0 < min_lam < max_lam, got min_lam={min_lam!r}, max_lam={max_lam!r}"
        )
    if not (radius > 0.0):
        raise ValueError(f"radius must be > 0, got {radius!r}")
    if not isinstance(max_order, (int, np.integer)) or int(max_order) < 2:
        raise ValueError(f"max_order must be an integer >= 2, got {max_order!r}")

    radius = float(radius)
    max_order = int(max_order)
    min_lam = float(min_lam)
    max_lam = float(max_lam)

    pre = _precompute(max_order)
    kz_even = pre["kz_even"]
    kz_odd = pre["kz_odd"]

    # j_{1,p} asymptotics ~ (p + 1/4) π; take enough roots that k_p <= max_lam.
    j_ceiling = max_lam * radius
    n_radial = max(1, int(np.ceil(j_ceiling / np.pi)) + 5)
    j1_roots = jn_zeros(1, n_radial)

    modes: list[tuple[float, int, int, float]] = []  # (lam, p, q, k)
    for p, j1 in enumerate(j1_roots, start=1):
        k = float(j1) / radius
        if k > max_lam:
            break
        for q in range(1, 2 * max_order + 1):
            lam = float(np.sqrt(k * k + (q * np.pi) ** 2))
            if min_lam <= lam <= max_lam:
                modes.append((lam, p, q, k))

    modes.sort(key=lambda t: t[0])
    n_modes = len(modes)

    an = np.zeros((n_modes, max_order), dtype=float)
    bn = np.zeros((n_modes, max_order), dtype=float)
    kr2_even = np.empty((n_modes, max_order), dtype=float)
    kr2_odd = np.empty((n_modes, max_order), dtype=float)
    lam_arr = np.empty(n_modes, dtype=float)
    min_bessel_denom = np.empty(n_modes, dtype=float)

    for i, (lam, _p, q, k) in enumerate(modes):
        lam_arr[i] = lam
        kr2_even[i] = lam * lam - kz_even**2
        kr2_odd[i] = lam * lam - kz_odd**2
        # Normalization denom for the selected oscillatory term: |J_0(k a)|.
        min_bessel_denom[i] = abs(float(jv(0, k * radius)))
        if q % 2 == 1:
            # q = 2n-1 → odd-integer cosine slot n = (q+1)/2 → index (q-1)//2.
            bn[i, (q - 1) // 2] = 1.0
        else:
            # q = 2n → even-integer sine slot n = q/2 → index q/2 - 1.
            an[i, q // 2 - 1] = 1.0

    return BeltramiSpectrum(
        m=0,
        radius=radius,
        max_order=max_order,
        lam=lam_arr,
        residual=np.zeros(n_modes, dtype=float),
        an=an,
        bn=bn,
        kr2_even=kr2_even,
        kr2_odd=kr2_odd,
        kz_even=kz_even.copy(),
        kz_odd=kz_odd.copy(),
        min_bessel_denom=min_bessel_denom,
        status=np.full(n_modes, int(IntervalStatus.ROOT), dtype=int),
        anharmonic=0.0,
    )


# ---------------------------------------------------------------------------
# JAX Chandrasekhar–Kendall field evaluators (Sec. III)
# ---------------------------------------------------------------------------


def _jax_jm_djm(jnp, bessel_jn, m: int, z):
    """``J_m(z)`` and ``J_m′(z)`` for integer ``m ≥ 0``.

    Uses ``bessel_jn(z, v=max(m, 1))`` which returns ``J_0…J_v`` stacked on
    axis 0. The recurrence ``J_m′ = J_{m−1} − (m/z) J_m`` is rewritten with
    a floor on ``|z|``; for ``m = 0`` the identity ``J_{-1} = −J_1`` gives
    ``J_0′ = −J_1``. The ``z → 0`` limit is patched explicitly because
    JAX ``bessel_jn`` returns NaN for ``|z| ≲ 2e-5``:

    * ``J_0 ≈ 1 − z²/4``, ``J_0′ ≈ −z/2``
    * ``J_1 ≈ z/2``, ``J_1′ ≈ 1/2``
    * ``J_m ≈ (z/2)^m / m!``, ``J_m′ ≈ 0`` for ``m ≥ 2``
    """
    v_max = max(m, 1)
    # Evaluate at a floored argument so bessel_jn never sees its NaN region;
    # the series branch below restores the correct small-z values.
    z_abs = jnp.abs(z)
    z_safe = jnp.maximum(z_abs, _BESSEL_JN_FLOOR)
    js = bessel_jn(z_safe, v=v_max)
    jm = js[m]
    if m == 0:
        # J_{-1} = -J_1 ⇒ J_0' = -J_1 (js[m-1] would wrap to js[-1] = J_1).
        jm_prev = -js[1]
    else:
        jm_prev = js[m - 1]
    djm = jm_prev - (m / z_safe) * jm
    near_zero = z_abs < _BESSEL_JN_FLOOR
    if m == 0:
        jm_lim = 1.0 - 0.25 * z * z
        djm_lim = -0.5 * z
    elif m == 1:
        jm_lim = 0.5 * z
        djm_lim = 0.5
    else:
        # Leading small-z term (z/2)^m / m! is O(z^m); treat both as 0.
        jm_lim = 0.0
        djm_lim = 0.0
    return (
        jnp.where(near_zero, jm_lim, jm),
        jnp.where(near_zero, djm_lim, djm),
    )


def _jax_ive_order(jnp, i0e, i1e, m: int, x):
    """Exponentially scaled ``I_m(x)`` from ``i0e`` / ``i1e`` plus recurrence.

    JAX exposes only ``I_0e`` and ``I_1e``. The upward recurrence

        I_{n+1}(x) = I_{n−1}(x) − (2n/x) I_n(x)

    is applied in a Python ``for`` over the fixed integer ``m`` (unrolled at
    trace time). ``x`` is floored so the ``1/x`` factor stays finite at the
    axis.
    """
    val_prev = i0e(x)
    if m == 0:
        return val_prev
    val_curr = i1e(x)
    if m == 1:
        return val_curr
    x_safe = jnp.maximum(x, _R_FLOOR)
    # Fixed-length recurrence: n runs 1 … m−1, producing I_m from I_0, I_1.
    for n in range(1, m):
        val_next = val_prev - (2.0 * n / x_safe) * val_curr
        val_prev, val_curr = val_curr, val_next
    return val_curr


def _jax_radial_rho(jnp, bessel_jn, i0e, i1e, m: int, radius: float, kr2, r):
    """Normalized radial profile ``ρ(R)`` and ``dρ/dR`` for every series term.

    Implements the Sec. IV normalization ``J_m(k R)/J_m(k a)`` (oscillatory),
    the scaled ``I_m`` analogue (evanescent), and the ``(R/a)^m`` power-law
    limit at vanishing radial wavenumber.

    Shape contract
    --------------
    ``kr2`` has shape ``(N,)`` or ``(M, N)``; ``r`` has shape ``(...)``.
    ``r`` is reshaped to ``r.shape + (1,) * kr2.ndim`` and ``kr2`` to
    ``(1,) * r.ndim + kr2.shape`` so every field point broadcasts against
    every mode/term. Returns ``rho, drho`` with shape ``r.shape + kr2.shape``
    (``(..., N)`` when ``kr2`` is 1-D; ``(..., M, N)`` when 2-D).

    JAX note
    --------
    All three branches are evaluated as fixed-shape expressions; ``jnp.where``
    then selects oscillatory / evanescent / power-law values elementwise.
    There is no data-dependent Python branching on ``kr2``.
    """
    # Append mode/Fourier axes: r (...) → r_col (... + 1*kr2.ndim).
    r_col = r.reshape(r.shape + (1,) * kr2.ndim)
    # Prepend field-point axes: kr2 (N,) or (M, N) → (1,…,1) + kr2.shape.
    kr2 = kr2.reshape((1,) * r.ndim + kr2.shape)
    pos = kr2 > _KR2_EPS
    neg = kr2 < -_KR2_EPS

    # Always compute both √(max(k²,0)) and √(max(−k²,0)); unused branch is
    # zeroed by the jnp.where selection below.
    k = jnp.sqrt(jnp.maximum(kr2, 0.0))
    kappa = jnp.sqrt(jnp.maximum(-kr2, 0.0))

    # Oscillatory branch: J_m(k R) / J_m(k a) and its R-derivative.
    jm_r, djm_r = _jax_jm_djm(jnp, bessel_jn, m, k * r_col)
    jm_a, _ = _jax_jm_djm(jnp, bessel_jn, m, k * radius)
    jm_a_safe = jnp.where(pos, jm_a, jnp.ones_like(jm_a))
    rho_j = jnp.where(pos, jm_r / jm_a_safe, 0.0)
    drho_j = jnp.where(pos, k * djm_r / jm_a_safe, 0.0)

    # Evanescent branch: I_m(κ R)/I_m(κ a) · exp(κ(R−a)) via scaled ive,
    # so the common exp(κ a) factor cancels in the ratio and the remaining
    # exp(κ(R−a)) is written explicitly for autodiff-friendly scaling.
    # For m = 0, _jax_ive_order(..., m-1 = -1, ...) skips the m == 0 / m == 1
    # early returns, then ``range(1, -1)`` is empty, so it returns I_1e —
    # which is correct because I_{-1} = I_1.
    x_r = kappa * r_col
    x_a = kappa * radius
    ive_m_r = _jax_ive_order(jnp, i0e, i1e, m, x_r)
    ive_m_a = _jax_ive_order(jnp, i0e, i1e, m, x_a)
    ive_mm1_r = _jax_ive_order(jnp, i0e, i1e, m - 1, x_r)
    scale = jnp.exp(kappa * (r_col - radius))
    ive_m_a_safe = jnp.where(neg, ive_m_a, jnp.ones_like(ive_m_a))
    rho_i = jnp.where(neg, ive_m_r / ive_m_a_safe * scale, 0.0)
    x_r_safe = jnp.maximum(x_r, _R_FLOOR)
    dim_over_ia = (ive_mm1_r - (m / x_r_safe) * ive_m_r) / ive_m_a_safe
    drho_i = jnp.where(neg, kappa * dim_over_ia * scale, 0.0)

    # Vanishing-k² branch: (R/a)^m and its derivative.
    # At m = 0 the power is identically 1 and the derivative is 0; the
    # expression (m/radius)*ratio**(m-1) is 0*inf = NaN at r = 0 and would
    # poison unselected zero-coefficient Fourier slots.
    ratio = r_col / radius
    rho_p = ratio**m
    if m == 0:
        drho_p = jnp.zeros_like(ratio)
    else:
        drho_p = (m / radius) * ratio ** (m - 1)

    # Elementwise branch select (all expressions already evaluated).
    rho = jnp.where(pos, rho_j, jnp.where(neg, rho_i, rho_p))
    drho = jnp.where(pos, drho_j, jnp.where(neg, drho_i, drho_p))
    return rho, drho


def _build_weighted_field(spectrum: BeltramiSpectrum, *, real: bool = False):
    """Internal jitted ``(r, phi, z, weights) -> (..., 3)`` for one spectrum.

    Contracts the mode axis inside the jit: spectrum arrays are stored as
    ``(M,)`` / ``(M, N)`` and the output is ``sum_j w_j B_j``. Shared by
    ``generate_beltrami_weighted_callable`` and the per-mode path in
    ``generate_beltrami_callable`` (one-mode slices with ``weights = [1]``).
    """
    import jax
    from jax.scipy.special import bessel_jn, i0e, i1e

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    m = int(spectrum.m)
    radius = float(spectrum.radius)
    anh = float(spectrum.anharmonic)
    # (M,), (M, N), (N,) — mode axis is contracted by the weight sum.
    lam = jnp.asarray(spectrum.lam, dtype=jnp.float64)
    an = jnp.asarray(spectrum.an, dtype=jnp.float64)
    bn = jnp.asarray(spectrum.bn, dtype=jnp.float64)
    kr2_even = jnp.asarray(spectrum.kr2_even, dtype=jnp.float64)
    kr2_odd = jnp.asarray(spectrum.kr2_odd, dtype=jnp.float64)
    # (1, N) so trig arrays are (..., 1, N) and broadcast over modes.
    kz_even = jnp.asarray(spectrum.kz_even, dtype=jnp.float64)[None, :]
    kz_odd = jnp.asarray(spectrum.kz_odd, dtype=jnp.float64)[None, :]
    n_modes = int(spectrum.n_modes)

    def field(r, phi, z, weights):
        r, phi, z = jnp.broadcast_arrays(
            jnp.asarray(r, dtype=jnp.float64),
            jnp.asarray(phi, dtype=jnp.float64),
            jnp.asarray(z, dtype=jnp.float64),
        )
        weights = jnp.asarray(weights, dtype=jnp.float64)
        # (..., 1) for broadcasting against the mode axis.
        r_safe = jnp.maximum(r, _R_FLOOR)[..., None]
        ratio = (r / radius)[..., None]
        power = ratio**m
        if m == 0:
            dpower = jnp.zeros_like(ratio)
        else:
            dpower = (m / radius) * ratio ** (m - 1)
        # phase_z: (..., M) from z (...,) and lam (M,).
        phase_z = jnp.exp(1j * lam * z[..., None])

        chi = anh * power * phase_z
        chi_r = anh * dpower * phase_z
        chi_z = anh * (1j * lam) * power * phase_z
        chi_rz = anh * (1j * lam) * dpower * phase_z
        chi_zz = anh * (-(lam**2)) * power * phase_z

        # Radial profiles: (..., M, N).
        rho_o, drho_o = _jax_radial_rho(
            jnp, bessel_jn, i0e, i1e, m, radius, kr2_odd, r
        )
        rho_e, drho_e = _jax_radial_rho(
            jnp, bessel_jn, i0e, i1e, m, radius, kr2_even, r
        )
        # z[..., None, None] → (..., 1, 1); kz_* is (1, N) → trig (..., 1, N).
        z_col = z[..., None, None]
        cos_o = jnp.cos(kz_odd * z_col)
        sin_o = jnp.sin(kz_odd * z_col)
        sin_e = jnp.sin(kz_even * z_col)
        cos_e = jnp.cos(kz_even * z_col)

        # Odd-integer cosine series (b_n): contract Fourier axis -1 → (..., M).
        chi = chi + jnp.sum(bn * rho_o * cos_o, axis=-1)
        chi_r = chi_r + jnp.sum(bn * drho_o * cos_o, axis=-1)
        chi_z = chi_z + jnp.sum(bn * rho_o * (-kz_odd) * sin_o, axis=-1)
        chi_rz = chi_rz + jnp.sum(bn * drho_o * (-kz_odd) * sin_o, axis=-1)
        chi_zz = chi_zz + jnp.sum(bn * rho_o * (-(kz_odd**2)) * cos_o, axis=-1)

        # Even-integer sine series (i a_n).
        chi = chi + 1j * jnp.sum(an * rho_e * sin_e, axis=-1)
        chi_r = chi_r + 1j * jnp.sum(an * drho_e * sin_e, axis=-1)
        chi_z = chi_z + 1j * jnp.sum(an * rho_e * kz_even * cos_e, axis=-1)
        chi_rz = chi_rz + 1j * jnp.sum(an * drho_e * kz_even * cos_e, axis=-1)
        chi_zz = chi_zz + 1j * jnp.sum(
            an * rho_e * (-(kz_even**2)) * sin_e, axis=-1
        )

        # Sec. III CK cylindrical components, then Cartesian projection.
        # phase_phi / cos_p / sin_p: (..., 1) against mode axis.
        phase_phi = jnp.exp(1j * m * phi)[..., None]
        b_r = (1j * m / r_safe * chi + chi_rz / lam) * phase_phi
        b_phi = (1j * m / (lam * r_safe) * chi_z - chi_r) * phase_phi
        b_z = (lam * chi + chi_zz / lam) * phase_phi

        cos_p = jnp.cos(phi)[..., None]
        sin_p = jnp.sin(phi)[..., None]
        b_x = b_r * cos_p - b_phi * sin_p
        b_y = b_r * sin_p + b_phi * cos_p
        # (..., M, 3) → weighted sum over modes → (..., 3).
        out = jnp.stack([b_x, b_y, b_z], axis=-1)
        out = jnp.sum(
            out * weights.reshape((1,) * (out.ndim - 2) + (n_modes, 1)),
            axis=-2,
        )
        return jnp.real(out) if real else out

    return jax.jit(field)


def generate_beltrami_weighted_callable(
    spectrum: BeltramiSpectrum, *, real: bool = False
) -> Callable:
    """Build one JAX-jitted weighted-sum field for an entire spectrum.

    Returns ``(r, phi, z, weights) -> B_xyz`` with shape ``(..., 3)``, where
    ``weights`` has length ``spectrum.n_modes`` and the mode axis is
    contracted inside the jit (one graph per ``m``, not per mode). With
    ``real=True`` the output is ``float64`` ``Re(Σ w_j B_j)``.
    """
    if not isinstance(spectrum, BeltramiSpectrum):
        raise TypeError(
            "generate_beltrami_weighted_callable expects a BeltramiSpectrum, "
            f"got {type(spectrum)!r}"
        )
    return _build_weighted_field(spectrum, real=real)


def generate_beltrami_callable(
    spectrum: BeltramiSpectrum, *, real: bool = False
) -> list[Callable]:
    """Build one JAX-jitted Cartesian field per spectrum mode.

    Reconstructs the Sec. III Chandrasekhar–Kendall field from the Sec. IV
    series stored in ``spectrum`` (also used for Sec. II axisymmetric modes
    encoded as one-hot ``an`` / ``bn`` with ``anharmonic = 0``). Each callable
    maps broadcastable ``(r, phi, z)`` to ``B_xyz`` with shape ``(..., 3)``.
    By default the output is ``complex128`` — the full complex field, not
    its real part. With ``real=True`` the callable returns ``float64``
    ``Re(B)``, which is itself a Beltrami field (curl and λ are real).
    Overall amplitude is set by ``spectrum.anharmonic`` on the Sec. IV seed
    plus the ``an`` / ``bn`` coefficients (paper convention: anharmonic
    coefficient 1 for ``m >= 1``).

    Each per-mode callable is a thin wrapper around the shared batched
    field body (one-mode spectrum slice, weights frozen to ``[1.0]``), so
    there is a single implementation of the CK formula.
    """
    from dataclasses import replace

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    if not isinstance(spectrum, BeltramiSpectrum):
        raise TypeError(
            f"generate_beltrami_callable expects a BeltramiSpectrum, got {type(spectrum)!r}"
        )

    callables: list[Callable] = []
    ones = jnp.asarray([1.0], dtype=jnp.float64)
    for i in range(spectrum.n_modes):
        # One-mode slice: same batched body, M=1, no batching overhead.
        slice_spec = replace(
            spectrum,
            lam=np.asarray(spectrum.lam[i : i + 1], dtype=float),
            residual=np.asarray(spectrum.residual[i : i + 1], dtype=float),
            an=np.asarray(spectrum.an[i : i + 1], dtype=float),
            bn=np.asarray(spectrum.bn[i : i + 1], dtype=float),
            kr2_even=np.asarray(spectrum.kr2_even[i : i + 1], dtype=float),
            kr2_odd=np.asarray(spectrum.kr2_odd[i : i + 1], dtype=float),
            min_bessel_denom=np.asarray(
                spectrum.min_bessel_denom[i : i + 1], dtype=float
            ),
        )
        weighted = _build_weighted_field(slice_spec, real=real)

        def field(r, phi, z, _fn=weighted, _w=ones):
            return _fn(r, phi, z, _w)

        callables.append(jax.jit(field))
    return callables
