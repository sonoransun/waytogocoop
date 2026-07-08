"""Curved graphene sheets: Monge-gauge strain and pseudo-magnetic fields.

Established physics unless marked SPECULATIVE.

An out-of-plane height field h(x, y) strains the graphene sheet; in the
Monge gauge (no in-plane relaxation) the strain tensor is
eps_ij = (1/2) d_i h d_j h.  Strain enters the Dirac Hamiltonian as a gauge
field, producing a pseudo-magnetic field B = curl A that can reach hundreds
of Tesla in nanoscale bubbles.  The x axis is taken along the zigzag
direction, which fixes the form of the strain gauge field.

CAVEAT — pseudo-field versus real field: the pseudo-magnetic field is
valley-antisymmetric (+B at K, -B at K') and preserves time-reversal
symmetry, so it does NOT orbitally depair singlet Cooper pairs the way a
real magnetic field does.  The ``gap_suppression_factor`` ansatz below is a
qualitative SPECULATIVE knob, not established pair-breaking physics.

Literature basis:
  - Vozmediano, Katsnelson & Guinea, Phys. Rep. 496, 109 (2010) — gauge
    fields in graphene, beta = -dln(t)/dln(a) ~ 2-3.4.
  - Guinea, Katsnelson & Geim, Nat. Phys. 6, 30 (2010) — strain
    engineering of uniform pseudo-fields.
  - Levy et al., Science 329, 544 (2010) — >300 T pseudo-fields observed
    in graphene nanobubbles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waytogocoop.config import (
    ANGSTROM_TO_M,
    B_PAIRBREAK_DEFAULT,
    BEND_RADIUS_DEFAULT,
    BUMP_HEIGHT_DEFAULT,
    BUMP_SIGMA_DEFAULT,
    ELEMENTARY_CHARGE_C,
    GRAPHENE_A_CC,
    GRAPHENE_BETA,
    HBAR_J_S,
    RIPPLE_WAVELENGTH_DEFAULT,
)

CURVATURE_GEOMETRIES: tuple[str, ...] = (
    "flat",
    "gaussian_bump",
    "sinusoidal_ripple",
    "cylindrical_bend",
    "spherical_cap",
)


@dataclass
class CurvatureConfig:
    """Configuration for a curved-sheet geometry."""

    geometry: str = "flat"                        # one of CURVATURE_GEOMETRIES
    amplitude: float = BUMP_HEIGHT_DEFAULT        # Angstrom — bump/ripple height
    sigma: float = BUMP_SIGMA_DEFAULT             # Angstrom — Gaussian bump width
    wavelength: float = RIPPLE_WAVELENGTH_DEFAULT  # Angstrom — ripple wavelength
    radius: float = BEND_RADIUS_DEFAULT           # Angstrom — bend/cap radius
    orientation_deg: float = 0.0                  # measured from zigzag = x axis
    beta: float = GRAPHENE_BETA                   # electron-phonon coupling -dln(t)/dln(a)
    b_pairbreak: float = B_PAIRBREAK_DEFAULT      # Tesla — SPECULATIVE pair-breaking scale
    valley: int = 1                               # +1 = K, -1 = K' (flips the pseudo-field)


@dataclass
class CurvatureResult:
    """Result of a curved-sheet pseudo-magnetic-field computation."""

    height: np.ndarray          # 2D h(x, y) in Angstrom
    strain_xx: np.ndarray       # 2D dimensionless Monge-gauge strain
    strain_yy: np.ndarray       # 2D dimensionless Monge-gauge strain
    strain_xy: np.ndarray       # 2D dimensionless Monge-gauge strain
    pseudo_field: np.ndarray    # 2D signed pseudo-field (Tesla, at the config's valley)
    gap_suppression: np.ndarray  # 2D factor in [0, 1] — SPECULATIVE
    max_abs_field: float        # Tesla


def height_field(config: CurvatureConfig, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluate the out-of-plane height field h(x, y) for a geometry.

    Parameters
    ----------
    config : CurvatureConfig
        Geometry name and shape parameters.
    x, y : np.ndarray
        1D arrays defining the grid axes (Angstrom).

    Returns
    -------
    np.ndarray
        Shape ``(len(y), len(x))`` height field (Angstrom).

    Notes
    -----
    The spherical cap uses ``h = sqrt(max(R^2 - r^2, 0))``; it covers the
    whole window when ``R`` is large compared to the window extent (the
    intended regime — the physics depends only on height gradients, so the
    constant offset ~R is irrelevant).
    """
    X, Y = np.meshgrid(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    phi = np.radians(config.orientation_deg)
    if config.geometry == "flat":
        return np.zeros_like(X)
    if config.geometry == "gaussian_bump":
        return config.amplitude * np.exp(-(X**2 + Y**2) / (2.0 * config.sigma**2))
    if config.geometry == "sinusoidal_ripple":
        u = X * np.cos(phi) + Y * np.sin(phi)
        return config.amplitude * np.sin(2.0 * np.pi * u / config.wavelength)
    if config.geometry == "cylindrical_bend":
        v = X * np.cos(phi) + Y * np.sin(phi)
        r = config.radius
        return r - np.sqrt(r**2 - np.minimum(np.abs(v), r) ** 2)
    if config.geometry == "spherical_cap":
        r = config.radius
        return np.sqrt(np.maximum(r**2 - (X**2 + Y**2), 0.0))
    raise ValueError(
        f"Unknown geometry '{config.geometry}'. Available: {list(CURVATURE_GEOMETRIES)}"
    )


def strain_tensor(
    height: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monge-gauge strain tensor of a height field.

    .. math::

        \\epsilon_{ij} = \\frac{1}{2}
            \\frac{\\partial h}{\\partial x_i}
            \\frac{\\partial h}{\\partial x_j}

    No in-plane relaxation is included.

    Parameters
    ----------
    height : np.ndarray
        2D height field, shape ``(len(y), len(x))`` (Angstrom).
    x, y : np.ndarray
        1D grid axes (Angstrom), uniformly spaced.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(eps_xx, eps_yy, eps_xy)``, dimensionless, same shape as *height*.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # np.gradient axis order: axis 0 is y (rows), so the first return is dh/dy
    hy, hx = np.gradient(height, dy, dx)
    return 0.5 * hx * hx, 0.5 * hy * hy, 0.5 * hx * hy


def pseudo_magnetic_field(
    eps_xx: np.ndarray,
    eps_yy: np.ndarray,
    eps_xy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    beta: float = GRAPHENE_BETA,
    valley: int = 1,
) -> np.ndarray:
    """Pseudo-magnetic field of a strained sheet at the chosen valley.

    With x along the zigzag direction the strain gauge field is

    .. math::

        \\mathbf{A} = \\frac{\\hbar \\beta}{2 e a_{cc}}
            \\begin{pmatrix} \\epsilon_{xx} - \\epsilon_{yy} \\\\
                             -2\\epsilon_{xy} \\end{pmatrix},
        \\qquad B = \\xi \\left(\\partial_x A_y - \\partial_y A_x\\right)

    Parameters
    ----------
    eps_xx, eps_yy, eps_xy : np.ndarray
        2D dimensionless strain components, shape ``(len(y), len(x))``.
    x, y : np.ndarray
        1D grid axes (Angstrom), uniformly spaced.
    beta : float
        Electron-phonon coupling -dln(t)/dln(a).
    valley : int
        Valley index :math:`\\xi`: +1 (K) or -1 (K').  The pseudo-field is
        valley-antisymmetric, so the returned field is multiplied by it.

    Returns
    -------
    np.ndarray
        Signed pseudo-field in Tesla at the chosen valley.

    Notes
    -----
    The 2-pixel border is clamped to the nearest interior values: the
    first-order one-sided differences np.gradient uses at the edges double
    up when the curl is taken, creating spurious border fields up to ~100x
    the interior signal (0.88 T spurious vs 0.008 T true for the default
    spherical cap).
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    prefactor = HBAR_J_S * beta / (2.0 * ELEMENTARY_CHARGE_C * GRAPHENE_A_CC * ANGSTROM_TO_M)
    a_x = prefactor * (eps_xx - eps_yy)
    a_y = prefactor * (-2.0 * eps_xy)
    # Gradients are per Angstrom; dividing by ANGSTROM_TO_M converts the
    # curl to per metre, giving Tesla for A in T*m.
    b_field = (
        np.gradient(a_y, dx, axis=1) - np.gradient(a_x, dy, axis=0)
    ) / ANGSTROM_TO_M
    b_field[0, :] = b_field[2, :]
    b_field[1, :] = b_field[2, :]
    b_field[-1, :] = b_field[-3, :]
    b_field[-2, :] = b_field[-3, :]
    b_field[:, 0] = b_field[:, 2]
    b_field[:, 1] = b_field[:, 2]
    b_field[:, -1] = b_field[:, -3]
    b_field[:, -2] = b_field[:, -3]
    return b_field * valley


def gap_suppression_factor(
    pseudo_field: np.ndarray, b_pairbreak: float = B_PAIRBREAK_DEFAULT
) -> np.ndarray:
    """SPECULATIVE: local gap-suppression factor from the pseudo-field.

    Linear pair-breaking ansatz clipped to [0, 1]:

    .. math::

        S(\\mathbf{r}) = \\max\\!\\left(0,\\;
            1 - \\frac{|B_s(\\mathbf{r})|}{B_{pb}}\\right)

    The pseudo-field preserves time-reversal symmetry and does not depair
    singlet pairs like a real field (see module caveat) — this is a
    qualitative visualization knob only.  Because only :math:`|B_s|` enters,
    the factor is valley-independent (the K and K' fields differ by sign).

    Parameters
    ----------
    pseudo_field : np.ndarray
        Signed pseudo-field (Tesla).
    b_pairbreak : float
        Field scale at which the gap is fully suppressed (Tesla).

    Returns
    -------
    np.ndarray
        Factor in [0, 1], same shape as *pseudo_field*.
    """
    return np.clip(1.0 - np.abs(pseudo_field) / b_pairbreak, 0.0, 1.0)


def displacement_field(
    height: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """SPECULATIVE: leading-order in-plane projected displacement of an
    inextensible bent sheet, u = -1/2 * h * grad(h).

    Projecting an unstretched sheet bent into the height profile h(x, y)
    back onto the plane displaces material points inward toward maxima; to
    leading order in the slope the projected displacement is
    :math:`\\mathbf{u} = -\\tfrac{1}{2} h \\nabla h`.  Feed the result into
    the ``displacement`` argument of the graphene layer potentials to warp a
    moire pattern (curvature back-reaction).

    Parameters
    ----------
    height : np.ndarray
        2D height field, shape ``(len(y), len(x))`` (Angstrom).
    x, y : np.ndarray
        1D grid axes (Angstrom), uniformly spaced.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(u_x, u_y)`` in Angstrom, same shape as *height*.
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # np.gradient axis order: axis 0 is y (rows), so the first return is dh/dy
    hy, hx = np.gradient(height, dy, dx)
    return -0.5 * height * hx, -0.5 * height * hy


def compute_curvature_effects(
    config: CurvatureConfig, x: np.ndarray, y: np.ndarray
) -> CurvatureResult:
    """Full pipeline: height field -> strain -> pseudo-field -> gap suppression.

    Parameters
    ----------
    config : CurvatureConfig
        Geometry, shape parameters, coupling, and pair-breaking scale.
    x, y : np.ndarray
        1D grid axes (Angstrom), uniformly spaced, at least 8 points each
        (the border clamp needs interior pixels).

    Returns
    -------
    CurvatureResult
        All 2D fields share the shape ``(len(y), len(x))``.
    """
    if config.geometry not in CURVATURE_GEOMETRIES:
        raise ValueError(
            f"Unknown geometry '{config.geometry}'. Available: {list(CURVATURE_GEOMETRIES)}"
        )
    if config.amplitude < 0:
        raise ValueError("amplitude must be >= 0")
    if config.sigma <= 0:
        raise ValueError("sigma must be positive")
    if config.wavelength <= 0:
        raise ValueError("wavelength must be positive")
    if config.radius <= 0:
        raise ValueError("radius must be positive")
    if config.b_pairbreak <= 0:
        raise ValueError("b_pairbreak must be positive")
    if config.valley not in (1, -1):
        raise ValueError(f"valley must be +1 or -1, got {config.valley}")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 8 or len(y) < 8:
        raise ValueError("x and y must each have at least 8 points")

    height = height_field(config, x, y)
    eps_xx, eps_yy, eps_xy = strain_tensor(height, x, y)
    pseudo_field = pseudo_magnetic_field(
        eps_xx, eps_yy, eps_xy, x, y, beta=config.beta, valley=config.valley
    )
    gap_suppression = gap_suppression_factor(pseudo_field, b_pairbreak=config.b_pairbreak)

    return CurvatureResult(
        height=height,
        strain_xx=eps_xx,
        strain_yy=eps_yy,
        strain_xy=eps_xy,
        pseudo_field=pseudo_field,
        gap_suppression=gap_suppression,
        max_abs_field=float(np.abs(pseudo_field).max()),
    )
