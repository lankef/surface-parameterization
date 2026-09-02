import numpy as np
import matplotlib.pyplot as plt


def _closed_gamma(surf, close=True):
    """Return ``surf.gamma()`` as a NumPy array, optionally seam-closed."""
    gamma = np.asarray(surf.gamma())
    if close:
        # quadpoints are open grids in [0, 1); wrap once in theta (and in phi
        # only if the phi grid spans the whole torus) so the mesh has no seam.
        gamma = np.concatenate([gamma, gamma[:, :1]], axis=1)
        if np.isclose(float(surf.quadpoints_phi[-1] + surf.dphi), 1.0):
            gamma = np.concatenate([gamma, gamma[:1]], axis=0)
    return gamma


def plot_gamma(surf, close=True, ax=None, **kwargs):
    """Plot ``surf.gamma()`` of a ``SurfaceJAX`` as a 3D surface, equal aspect."""
    gamma = _closed_gamma(surf, close=close)

    if ax is None:
        ax = plt.figure(figsize=(7, 6)).add_subplot(projection='3d')
    x, y, z = gamma[..., 0], gamma[..., 1], gamma[..., 2]
    ax.plot_surface(
        x, y, z,
        **{'rstride': 1, 'cstride': 1, 'linewidth': 0,
           'antialiased': False, 'cmap': 'viridis', **kwargs},
    )
    ax.set_aspect('equal')  # matplotlib >= 3.6
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax


def to_vtk(surf, filename, close=True):
    """Save ``surf.gamma()`` as a VTK structured grid via ``pyevtk``.

    Uses the same seam-closing rules as :func:`plot_gamma`. ``filename`` may
    omit the extension; ``pyevtk`` writes ``filename.vts``.
    """
    try:
        from pyevtk.hl import gridToVTK
    except ImportError as exc:
        raise ImportError('pyevtk must be installed to save vtk files.') from exc

    gamma = _closed_gamma(surf, close=close)
    ntor, npol = gamma.shape[:2]
    # pyevtk expects shape (1, ntor, npol) contiguous arrays
    x = np.ascontiguousarray(gamma[:, :, 0].reshape(1, ntor, npol))
    y = np.ascontiguousarray(gamma[:, :, 1].reshape(1, ntor, npol))
    z = np.ascontiguousarray(gamma[:, :, 2].reshape(1, ntor, npol))
    return gridToVTK(str(filename), x, y, z)