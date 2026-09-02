#!/usr/bin/env python3
"""Fit an NCSX surface with evosax CMA-ES.

By default (``DUMMY=True``) optimizes ``SurfaceXYZFourierJAX`` dofs as a
sanity check that the surface-surface distance objective behaves. Set
``DUMMY=False`` to optimize ``SurfaceBeltramiJAX`` instead.

Run with the ``desc`` conda env (GPU JAX + evosax 0.2)::

    conda activate desc
    python optimize_beltrami_cma.py
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt import load

from evosax.algorithms import CMA_ES
from quadcoil.surface import SurfaceJAX, SurfaceXYZFourierJAX
from surface_distance import surface_surface_distance
from surface_transport import SurfaceBeltramiJAX
from utils import to_vtk

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

NCSX_PATH = Path("ncsx.json")
# When True, CMA-ES optimizes SurfaceXYZFourierJAX (fast objective check).
# When False, optimizes SurfaceBeltramiJAX.
DUMMY = True
OUTDIR = Path("runs/xyz_fourier_cma" if DUMMY else "runs/beltrami_cma")

POPSIZE = 200
N_GEN = 100
STD_INIT = 0.1
SEED = 0
# "vmap" batches the whole population; "map" evaluates one at a time
# (lower peak memory if vmap OOMs).
EVAL_MODE = "vmap"

N_PHI = 16
N_THETA = 16
QUAD_N = 16  # surface quadpoints per field period
# Dummy (XYZ Fourier) resolution.
DUMMY_MPOL = 4
DUMMY_NTOR = 4
# Beltrami-only knobs (ignored when DUMMY=True).
STEP_NUM = 10
M_PER_PERIOD = 3
MIN_LAM = 1e-5
MAX_LAM = 10.0
N_LAM = 1000
MAX_ORDER = 12
MAX_ITER = 100
TOL = 1e-5
PI_GUARD = 1e-3
POLISH = True
# Write VTK every K generations (0 = only at end).
SAVE_EVERY = 0


def _circular_torus_gamma(major: float, minor: float, phi: jnp.ndarray, theta: jnp.ndarray):
    """Axisymmetric circular torus on a (phi, theta) meshgrid (ij indexing)."""
    phi_rad = 2.0 * jnp.pi * phi
    theta_rad = 2.0 * jnp.pi * theta
    R = major + minor * jnp.cos(theta_rad)
    Z = minor * jnp.sin(theta_rad)
    X = R * jnp.cos(phi_rad)
    Y = R * jnp.sin(phi_rad)
    return jnp.stack([X, Y, Z], axis=-1)


def main() -> None:
    jax.config.update("jax_enable_x64", True)
    print(f"jax {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")
    print(f"DUMMY={DUMMY}  OUTDIR={OUTDIR}")

    OUTDIR.mkdir(parents=True, exist_ok=True)

    _, simsopt_surface = load(str(NCSX_PATH))
    target_surface = SurfaceJAX.from_simsopt(simsopt_surface)
    target_volume = simsopt_surface.volume()
    minor = float(
        np.sqrt(target_volume / (2 * np.pi**2) / simsopt_surface.major_radius())
    )
    major = float(simsopt_surface.major_radius())
    nfp = int(simsopt_surface.nfp)
    stellsym = bool(simsopt_surface.stellsym)
    print(f"NCSX nfp={nfp}  R0={major:.4f}  a={minor:.4f}")

    quad_phi = jnp.linspace(0.0, 1.0 / nfp, QUAD_N, endpoint=False)
    quad_theta = jnp.linspace(0.0, 1.0, QUAD_N, endpoint=False)

    if DUMMY:
        # Start from a circular torus in XYZ Fourier form — far from NCSX, so
        # a working objective should drive fitness down under CMA-ES.
        phi_mesh, theta_mesh = jnp.meshgrid(quad_phi, quad_theta, indexing="ij")
        gamma0 = _circular_torus_gamma(major, minor, phi_mesh, theta_mesh)
        init_surface = SurfaceXYZFourierJAX.fit(
            phi_target=phi_mesh,
            theta_target=theta_mesh,
            gamma_target=gamma0,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=quad_phi,
            quadpoints_theta=quad_theta,
            mpol=DUMMY_MPOL,
            ntor=DUMMY_NTOR,
        )
        n_dofs = int(init_surface.dofs.shape[0])
        print(
            f"dummy SurfaceXYZFourierJAX  n_dofs={n_dofs}  "
            f"mpol={DUMMY_MPOL}  ntor={DUMMY_NTOR}"
        )

        def make_surface(dofs):
            return SurfaceXYZFourierJAX(
                nfp=init_surface.nfp,
                stellsym=init_surface.stellsym,
                mpol=init_surface.mpol,
                ntor=init_surface.ntor,
                quadpoints_phi=init_surface.quadpoints_phi,
                quadpoints_theta=init_surface.quadpoints_theta,
                dofs=dofs,
            )
    else:
        box_r = major + 2.0 * minor
        box_z = 4.0 * minor
        init_surface = SurfaceBeltramiJAX(
            nfp=nfp,
            stellsym=stellsym,
            seed_major_radius=major,
            seed_minor_radius=minor,
            quadpoints_phi=quad_phi,
            quadpoints_theta=quad_theta,
            step_num=STEP_NUM,
            m_per_period=M_PER_PERIOD,
            min_lam=MIN_LAM,
            max_lam=MAX_LAM,
            n_lam=N_LAM,
            box_r=box_r,
            box_z=box_z,
            max_order=MAX_ORDER,
            max_iter=MAX_ITER,
            tol=TOL,
            pi_guard=PI_GUARD,
            polish=POLISH,
        )
        n_dofs = int(init_surface.dofs.shape[0])
        print(
            f"basis n_modes={n_dofs}  m={init_surface.basis.m.tolist()}  "
            f"radius={init_surface.basis.radius:.4f}  "
            f"step_num={init_surface.step_num}  max_order={init_surface.max_order}"
        )

        def make_surface(dofs):
            return SurfaceBeltramiJAX(
                nfp=init_surface.nfp,
                stellsym=init_surface.stellsym,
                seed_major_radius=init_surface.seed_major_radius,
                seed_minor_radius=init_surface.seed_minor_radius,
                quadpoints_phi=init_surface.quadpoints_phi,
                quadpoints_theta=init_surface.quadpoints_theta,
                step_num=init_surface.step_num,
                m_per_period=init_surface.m_per_period,
                min_lam=init_surface.min_lam,
                max_lam=init_surface.max_lam,
                n_lam=init_surface.n_lam,
                dofs=dofs,
                box_r=init_surface.box_r,
                box_z=init_surface.box_z,
                max_order=init_surface.max_order,
                max_iter=init_surface.max_iter,
                tol=init_surface.tol,
                pi_guard=init_surface.pi_guard,
                polish=init_surface.polish,
            )

    def fitness_one(dofs):
        return surface_surface_distance(
            target_surface, make_surface(dofs), N_PHI, N_THETA
        )

    f0 = float(fitness_one(jnp.asarray(init_surface.dofs)))
    print(f"initial fitness={f0:.6e}")

    if EVAL_MODE == "vmap":
        fitness_fn = jax.jit(jax.vmap(fitness_one))
    elif EVAL_MODE == "map":
        # Peak memory ~ one individual; still jitted once.
        fitness_fn = jax.jit(lambda pop: jax.lax.map(fitness_one, pop))
    else:
        raise ValueError(f"EVAL_MODE must be 'vmap' or 'map', got {EVAL_MODE!r}")

    mean0 = jnp.asarray(init_surface.dofs, dtype=jnp.float64)
    es = CMA_ES(population_size=POPSIZE, solution=jnp.zeros((n_dofs,)))
    params = es.default_params.replace(std_init=float(STD_INIT))
    key = jax.random.PRNGKey(SEED)
    key, key_init = jax.random.split(key)
    state = es.init(key_init, mean0, params)

    hist_gen = []
    hist_best = []
    hist_gen_best = []
    hist_std = []
    hist_time = []
    hist_best_dofs = []

    t_start = time.perf_counter()
    print(
        f"{'gen':>5}  {'best':>14}  {'gen_best':>14}  "
        f"{'|mean|':>10}  {'std':>10}  {'dt_s':>8}"
    )
    for gen in range(N_GEN):
        t_gen = time.perf_counter()
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = es.ask(key_ask, state, params)
        fitness = jax.block_until_ready(fitness_fn(population))
        state, metrics = es.tell(key_tell, population, fitness, state, params)

        best_f = float(state.best_fitness)
        gen_best = float(metrics["best_fitness_in_generation"])
        mean_norm = float(jnp.linalg.norm(state.mean))
        std = float(state.std)
        dt = time.perf_counter() - t_gen
        print(
            f"{gen:5d}  {best_f:14.6e}  {gen_best:14.6e}  "
            f"{mean_norm:10.4e}  {std:10.4e}  {dt:8.2f}"
        )

        hist_gen.append(gen)
        hist_best.append(best_f)
        hist_gen_best.append(gen_best)
        hist_std.append(std)
        hist_time.append(time.perf_counter() - t_start)
        # best_solution is stored raveled; unravel to the dofs vector.
        best_dofs = np.asarray(es._unravel_solution(state.best_solution))
        hist_best_dofs.append(best_dofs)

        if SAVE_EVERY > 0 and (gen + 1) % SAVE_EVERY == 0:
            surf = make_surface(jnp.asarray(best_dofs))
            to_vtk(surf, OUTDIR / f"gen_{gen:04d}")

    np.savez(
        OUTDIR / "history.npz",
        gen=np.asarray(hist_gen, dtype=np.int32),
        best_fitness=np.asarray(hist_best, dtype=np.float64),
        gen_best_fitness=np.asarray(hist_gen_best, dtype=np.float64),
        std=np.asarray(hist_std, dtype=np.float64),
        time_s=np.asarray(hist_time, dtype=np.float64),
        best_dofs=np.asarray(hist_best_dofs, dtype=np.float64),
    )

    best_surf = make_surface(jnp.asarray(hist_best_dofs[-1]))
    written = to_vtk(best_surf, OUTDIR / "best")
    print(f"Wrote {OUTDIR / 'history.npz'}")
    print(f"Wrote {written}")
    print(f"Final best fitness: {hist_best[-1]:.6e}  total time {hist_time[-1]:.1f}s")


if __name__ == "__main__":
    main()
