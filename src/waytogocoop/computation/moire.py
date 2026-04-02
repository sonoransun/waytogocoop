"""Moire superlattice pattern generation and periodicity calculations."""

from __future__ import annotations

import numpy as np

from waytogocoop.materials.lattice import apply_rotation


def moire_periodicity_1d(a1: float, a2: float) -> float:
    """Compute the 1D moire period from two lattice constants.

    .. math::

        L = \\frac{a_1 \\, a_2}{|a_1 - a_2|}

    Parameters
    ----------
    a1, a2 : float
        Lattice constants (Angstrom) of the two layers.

    Returns
    -------
    float
        Moire period in Angstrom.
    """
    return a1 * a2 / abs(a1 - a2)


def moire_periodicity_with_twist(a: float, theta_deg: float) -> float:
    """Compute the moire period for a homo-bilayer with twist angle.

    .. math::

        L = \\frac{a}{2 \\sin(\\theta / 2)}

    Parameters
    ----------
    a : float
        Lattice constant (Angstrom).
    theta_deg : float
        Twist angle in degrees.

    Returns
    -------
    float
        Moire period in Angstrom.
    """
    theta = np.radians(theta_deg)
    sin_half = np.sin(theta / 2.0)
    if abs(sin_half) < 1e-12:
        return np.inf
    return a / (2.0 * sin_half)


def _reciprocal_g_vectors_square(a: float) -> np.ndarray:
    """First-order reciprocal lattice vectors for a square lattice.

    Returns 4 vectors at (2pi/a, 0), (0, 2pi/a), (-2pi/a, 0), (0, -2pi/a).
    """
    g = 2.0 * np.pi / a
    return np.array([
        [g, 0.0],
        [0.0, g],
        [-g, 0.0],
        [0.0, -g],
    ])


def _reciprocal_g_vectors_hexagonal(a: float) -> np.ndarray:
    """First-order reciprocal lattice vectors for a hexagonal lattice.

    Returns 6 vectors at 60-degree intervals with magnitude 4*pi/(a*sqrt(3)).
    """
    g_mag = 4.0 * np.pi / (a * np.sqrt(3.0))
    angles = np.arange(6) * np.pi / 3.0  # 0, 60, 120, 180, 240, 300 deg
    gx = g_mag * np.cos(angles)
    gy = g_mag * np.sin(angles)
    return np.column_stack([gx, gy])


def _plane_wave_sum(g_vectors: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate V(r) = sum_i cos(G_i . r) on a 2D grid.

    Parameters
    ----------
    g_vectors : np.ndarray
        Shape ``(M, 2)`` reciprocal lattice vectors.
    x, y : np.ndarray
        1D arrays defining the grid axes.

    Returns
    -------
    np.ndarray
        Shape ``(len(y), len(x))`` potential field.
    """
    # Build 2D coordinate grids — shape (Ny, Nx)
    X, Y = np.meshgrid(x, y)
    result = np.zeros_like(X)
    for gvec in g_vectors:
        result += np.cos(gvec[0] * X + gvec[1] * Y)
    return result


def generate_moire_pattern(
    substrate_a: float,
    overlayer_a: float,
    overlayer_lattice_type: str,
    twist_angle_deg: float = 0.0,
    grid_size: int = 200,
    physical_extent: float = 100.0,
    dw_factor_substrate: float = 1.0,
    dw_factor_overlayer: float = 1.0,
) -> dict:
    """Generate a moire interference pattern for a substrate/overlayer pair.

    The pattern is computed as the product of plane-wave sums for each
    layer's first-order reciprocal lattice vectors, normalized to [0, 1].

    Parameters
    ----------
    substrate_a : float
        Substrate lattice constant (Angstrom).  Assumed square.
    overlayer_a : float
        Overlayer lattice constant (Angstrom).
    overlayer_lattice_type : str
        ``"hexagonal"`` or ``"square"``.
    twist_angle_deg : float
        In-plane twist angle applied to the overlayer (degrees).
    grid_size : int
        Number of grid points per axis.
    physical_extent : float
        Half-width of the real-space window (Angstrom).
    dw_factor_substrate : float
        Debye-Waller amplitude scaling for substrate potential (default 1.0).
    dw_factor_overlayer : float
        Debye-Waller amplitude scaling for overlayer potential (default 1.0).

    Returns
    -------
    dict
        ``x`` (1D ndarray), ``y`` (1D ndarray), ``pattern`` (2D ndarray
        shape ``(grid_size, grid_size)``), ``moire_period`` (float).
    """
    x = np.linspace(-physical_extent, physical_extent, grid_size)
    y = np.linspace(-physical_extent, physical_extent, grid_size)

    # Substrate G vectors (square)
    g_sub = _reciprocal_g_vectors_square(substrate_a)

    # Overlayer G vectors
    if overlayer_lattice_type == "hexagonal":
        g_over = _reciprocal_g_vectors_hexagonal(overlayer_a)
    else:
        g_over = _reciprocal_g_vectors_square(overlayer_a)

    # Apply twist to overlayer G vectors
    if abs(twist_angle_deg) > 1e-12:
        g_over = apply_rotation(g_over, twist_angle_deg)

    # Compute potentials
    v_sub = _plane_wave_sum(g_sub, x, y)
    v_over = _plane_wave_sum(g_over, x, y)

    # Moire pattern = product, normalized to [0, 1]
    pattern = (dw_factor_substrate * v_sub) * (dw_factor_overlayer * v_over)
    p_min = pattern.min()
    p_max = pattern.max()
    if p_max - p_min > 1e-12:
        pattern = (pattern - p_min) / (p_max - p_min)
    else:
        pattern = np.zeros_like(pattern)

    # Estimate moire period
    if abs(twist_angle_deg) > 1e-6:
        # Use twist formula with effective average lattice constant
        a_eff = (substrate_a + overlayer_a) / 2.0
        moire_period = moire_periodicity_with_twist(a_eff, twist_angle_deg)
    else:
        if abs(substrate_a - overlayer_a) < 1e-12:
            moire_period = np.inf
        else:
            moire_period = moire_periodicity_1d(substrate_a, overlayer_a)

    return {
        "x": x,
        "y": y,
        "pattern": pattern,
        "moire_period": moire_period,
    }
