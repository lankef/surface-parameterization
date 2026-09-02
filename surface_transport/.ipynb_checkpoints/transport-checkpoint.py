from functools import partial

import jax.numpy as jnp
from jax import jacfwd, jit, lax, tree_util, vmap

from quadcoil.surface import SurfaceJAX, SurfaceRZFourierJAX


@tree_util.register_pytree_node_class
class SurfaceTransportJAX(SurfaceJAX):
    """Toroidal surface obtained by transporting a circular RZFourier seed.

    Subclasses implement :meth:`_gamma_map_single`. Geometry (normals, area,
    etc.) is inherited from :class:`SurfaceJAX` via :meth:`gammadash_at_point`.
    """

    def __init__(
        self,
        nfp,
        stellsym,
        seed_major_radius,
        seed_minor_radius,
        quadpoints_phi,
        quadpoints_theta,
    ):
        dofs = jnp.asarray([
            seed_major_radius,
            seed_minor_radius,
            seed_minor_radius,
        ])
        self.seed_major_radius = seed_major_radius
        self.seed_minor_radius = seed_minor_radius
        self.seed_surf = SurfaceRZFourierJAX(
            nfp=nfp,
            stellsym=stellsym,
            mpol=1,
            ntor=0,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=dofs,
        )
        super().__init__(
            nfp=nfp,
            stellsym=stellsym,
            mpol=1,
            ntor=0,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=dofs,
        )

    def _gamma_map_single(self, gamma_init):
        """Map a single seed point ``gamma_init`` of shape ``(3,)`` to ``(3,)``."""
        raise NotImplementedError(
            f'_gamma_map_single is is not implemented for {type(self)}'
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
        return type(self)(
            nfp=self.nfp,
            stellsym=self.stellsym,
            seed_major_radius=self.seed_major_radius,
            seed_minor_radius=self.seed_minor_radius,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

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
        children = (
            self.seed_major_radius,
            self.seed_minor_radius,
            self.quadpoints_phi,
            self.quadpoints_theta,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            nfp=aux_data['nfp'],
            stellsym=aux_data['stellsym'],
            seed_major_radius=children[0],
            seed_minor_radius=children[1],
            quadpoints_phi=children[2],
            quadpoints_theta=children[3],
        )


@tree_util.register_pytree_node_class
class SurfaceFlowJAX(SurfaceTransportJAX):
    """Surface transported by integrating an abstract flow with RK4."""

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

    def copy_and_set_quadpoints(self, quadpoints_phi, quadpoints_theta):
        return type(self)(
            nfp=self.nfp,
            stellsym=self.stellsym,
            seed_major_radius=self.seed_major_radius,
            seed_minor_radius=self.seed_minor_radius,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            step_num=self.step_num,
        )

    def tree_flatten(self):
        children = (
            self.seed_major_radius,
            self.seed_minor_radius,
            self.quadpoints_phi,
            self.quadpoints_theta,
        )
        aux_data = {
            'nfp': self.nfp,
            'stellsym': self.stellsym,
            'step_num': self.step_num,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            nfp=aux_data['nfp'],
            stellsym=aux_data['stellsym'],
            seed_major_radius=children[0],
            seed_minor_radius=children[1],
            quadpoints_phi=children[2],
            quadpoints_theta=children[3],
            step_num=aux_data['step_num'],
        )
