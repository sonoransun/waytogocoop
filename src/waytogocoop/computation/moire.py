"""Moire superlattice pattern generation and periodicity calculations."""

from __future__ import annotations

import numpy as np

from waytogocoop.config import SMALL_ANGLE_THRESHOLD, ZERO_THRESHOLD
from waytogocoop.materials.lattice import apply_rotation


def moire_periodicity_1d(a1, a2):
    """Compute the 1D moire period from two lattice constants.

    .. math::

        L = \\frac{a_1 \\, a_2}{|a_1 - a_2|}

    Parameters
    ----------
    a1, a2 : float or array_like
        Lattice constants (Angstrom) of the two layers.

    Returns
    -------
    float or np.ndarray
        Moire period in Angstrom.
    """
    if np.any(np.asarray(a1) <= 0) or np.any(np.asarray(a2) <= 0):
        raise ValueError("Lattice constants a1 and a2 must be positive")
    a2 = np.asarray(a2, dtype=float)
    diff = np.abs(a1 - a2)
    safe_diff = np.where(diff < ZERO_THRESHOLD, 1.0, diff)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(diff < ZERO_THRESHOLD, np.inf, a1 * a2 / safe_diff)
    return float(result) if result.ndim == 0 else result


def moire_periodicity_with_twist(a, theta_deg):
    """Compute the moire period for a homo-bilayer with twist angle.

    .. math::

        L = \\frac{a}{2 \\sin(\\theta / 2)}

    Parameters
    ----------
    a : float
        Lattice constant (Angstrom).
    theta_deg : float or array_like
        Twist angle in degrees.

    Returns
    -------
    float or np.ndarray
        Moire period in Angstrom.
    """
    if a <= 0:
        raise ValueError("Lattice constant a must be positive")
    theta_deg = np.asarray(theta_deg, dtype=float)
    theta = np.radians(theta_deg)
    sin_half = np.sin(theta / 2.0)
    abs_sin = np.abs(sin_half)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(abs_sin < ZERO_THRESHOLD, np.inf, a / (2.0 * abs_sin))
    return float(result) if result.ndim == 0 else result


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
    substrate_lattice_type: str = "square",
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
        Substrate lattice constant (Angstrom).
    overlayer_a : float
        Overlayer lattice constant (Angstrom).
    overlayer_lattice_type : str
        ``"hexagonal"`` or ``"square"``.
    substrate_lattice_type : str
        ``"hexagonal"`` or ``"square"``.  Defaults to ``"square"`` for
        backwards compatibility with FeTe-substrate callers.
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

    Notes
    -----
    The moire period reported when both ``substrate_a != overlayer_a`` *and*
    ``twist_angle_deg != 0`` uses ``a_eff = (substrate_a + overlayer_a) / 2``
    as the twist-formula input; this is only physically meaningful when the
    two lattice constants are close (e.g. a homo-bilayer).
    """
    if substrate_a <= 0:
        raise ValueError("substrate_a must be positive")
    if overlayer_a <= 0:
        raise ValueError("overlayer_a must be positive")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if physical_extent <= 0:
        raise ValueError("physical_extent must be positive")
    if overlayer_lattice_type not in ("hexagonal", "square"):
        raise ValueError(
            f"overlayer_lattice_type must be 'hexagonal' or 'square', "
            f"got '{overlayer_lattice_type}'"
        )
    if substrate_lattice_type not in ("hexagonal", "square"):
        raise ValueError(
            f"substrate_lattice_type must be 'hexagonal' or 'square', "
            f"got '{substrate_lattice_type}'"
        )

    x = np.linspace(-physical_extent, physical_extent, grid_size)
    y = np.linspace(-physical_extent, physical_extent, grid_size)

    # Substrate G vectors
    if substrate_lattice_type == "hexagonal":
        g_sub = _reciprocal_g_vectors_hexagonal(substrate_a)
    else:
        g_sub = _reciprocal_g_vectors_square(substrate_a)

    # Overlayer G vectors
    if overlayer_lattice_type == "hexagonal":
        g_over = _reciprocal_g_vectors_hexagonal(overlayer_a)
    else:
        g_over = _reciprocal_g_vectors_square(overlayer_a)

    # Apply twist to overlayer G vectors
    if abs(twist_angle_deg) > ZERO_THRESHOLD:
        g_over = apply_rotation(g_over, twist_angle_deg)

    # Compute potentials
    v_sub = _plane_wave_sum(g_sub, x, y)
    v_over = _plane_wave_sum(g_over, x, y)

    # Moire pattern = product, normalized to [0, 1]
    pattern = (dw_factor_substrate * v_sub) * (dw_factor_overlayer * v_over)
    pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
    p_min = pattern.min()
    p_max = pattern.max()
    if p_max - p_min > ZERO_THRESHOLD:
        pattern = (pattern - p_min) / (p_max - p_min)
    else:
        pattern = np.zeros_like(pattern)

    # Estimate moire period
    if abs(twist_angle_deg) > SMALL_ANGLE_THRESHOLD:
        # Use twist formula with effective average lattice constant
        a_eff = (substrate_a + overlayer_a) / 2.0
        moire_period = moire_periodicity_with_twist(a_eff, twist_angle_deg)
    else:
        if abs(substrate_a - overlayer_a) < ZERO_THRESHOLD:
            moire_period = np.inf
        else:
            moire_period = moire_periodicity_1d(substrate_a, overlayer_a)

    return {
        "x": x,
        "y": y,
        "pattern": pattern,
        "moire_period": moire_period,
    }
