"""Lattice generation and reciprocal-space utilities."""

from __future__ import annotations

import numpy as np


def square_lattice(a: float, nx: int, ny: int) -> np.ndarray:
    """Generate atom positions on a square lattice.

    Parameters
    ----------
    a : float
        Lattice constant (Angstrom).
    nx, ny : int
        Number of unit cells along x and y.

    Returns
    -------
    np.ndarray
        Shape ``(nx*ny, 2)`` array of (x, y) positions.
    """
    if a <= 0:
        raise ValueError(f"Lattice constant 'a' must be positive, got {a}")
    if nx < 1:
        raise ValueError(f"nx must be >= 1, got {nx}")
    if ny < 1:
        raise ValueError(f"ny must be >= 1, got {ny}")
    ix = np.arange(nx)
    iy = np.arange(ny)
    gx, gy = np.meshgrid(ix, iy)
    positions = np.column_stack([gx.ravel(), gy.ravel()]) * a
    return positions


def hexagonal_lattice(a: float, nx: int, ny: int) -> np.ndarray:
    """Generate atom positions on a hexagonal (triangular) lattice.

    The primitive vectors are ``a1 = a*(1, 0)`` and
    ``a2 = a*(1/2, sqrt(3)/2)``.

    Parameters
    ----------
    a : float
        Lattice constant (Angstrom).
    nx, ny : int
        Number of unit cells along each primitive direction.

    Returns
    -------
    np.ndarray
        Shape ``(nx*ny, 2)`` array of (x, y) positions.
    """
    if a <= 0:
        raise ValueError(f"Lattice constant 'a' must be positive, got {a}")
    if nx < 1:
        raise ValueError(f"nx must be >= 1, got {nx}")
    if ny < 1:
        raise ValueError(f"ny must be >= 1, got {ny}")
    a1 = np.array([a, 0.0])
    a2 = np.array([a * 0.5, a * np.sqrt(3) / 2.0])
    in1 = np.arange(nx)
    in2 = np.arange(ny)
    g1, g2 = np.meshgrid(in1, in2)
    g1 = g1.ravel()
    g2 = g2.ravel()
    positions = np.outer(g1, a1) + np.outer(g2, a2)
    return positions


def apply_rotation(points: np.ndarray, theta_deg: float) -> np.ndarray:
    """Rotate a set of 2D points about the origin.

    Parameters
    ----------
    points : np.ndarray
        Shape ``(N, 2)`` array.
    theta_deg : float
        Rotation angle in degrees (counter-clockwise positive).

    Returns
    -------
    np.ndarray
        Rotated points, same shape as input.
    """
    theta = np.radians(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    return points @ rot.T


def lattice_vectors(lattice_type: str, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Return primitive lattice vectors for a given lattice type.

    Parameters
    ----------
    lattice_type : str
        ``"square"`` or ``"hexagonal"``.
    a : float
        Lattice constant (Angstrom).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Primitive vectors ``(a1, a2)``, each shape ``(2,)``.
    """
    if a <= 0:
        raise ValueError(f"Lattice constant 'a' must be positive, got {a}")
    if lattice_type == "square":
        a1 = np.array([a, 0.0])
        a2 = np.array([0.0, a])
    elif lattice_type == "hexagonal":
        a1 = np.array([a, 0.0])
        a2 = np.array([a * 0.5, a * np.sqrt(3) / 2.0])
    else:
        raise ValueError(f"Unknown lattice_type: {lattice_type!r}")
    return a1, a2


def reciprocal_vectors(a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute 2D reciprocal lattice vectors from real-space primitives.

    Uses the relation ``b_i . a_j = 2*pi*delta_{ij}``.

    Parameters
    ----------
    a1, a2 : np.ndarray
        Real-space primitive vectors, each shape ``(2,)``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Reciprocal vectors ``(b1, b2)``, each shape ``(2,)``.
    """
    # Cross product in 2D: a1 x a2 = a1[0]*a2[1] - a1[1]*a2[0]
    cross = a1[0] * a2[1] - a1[1] * a2[0]
    if abs(cross) < 1e-30:
        raise ValueError("Lattice vectors are parallel; cannot compute reciprocal vectors")
    # b1 = 2*pi * (a2_perp) / cross  where a2_perp = (a2[1], -a2[0])
    b1 = 2.0 * np.pi * np.array([a2[1], -a2[0]]) / cross
    b2 = 2.0 * np.pi * np.array([-a1[1], a1[0]]) / cross
    return b1, b2
