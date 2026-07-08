"""Twisted and stacked graphene: moire patterns and magic-angle flat-band superconductivity.

Established physics unless marked SPECULATIVE.

Stacking registries (AA, AB/Bernal, ABC/rhombohedral) are encoded as phase
offsets on the first-shell reciprocal lattice vectors, reusing the moire
module's private ``_reciprocal_g_vectors_hexagonal`` shell (same package,
private by convention).  Twist angles reuse ``apply_rotation`` and the
homo-bilayer periodicity formula from the moire module.

Literature basis:
  - Bistritzer & MacDonald, PNAS 108, 12233 (2011) — continuum model,
    first magic angle where the Dirac velocity vanishes.
  - Cao et al., Nature 556, 43 (2018) — superconductivity in magic-angle
    twisted bilayer graphene (Tc ~ 2 K near filling nu ~ 2.4).
  - Park et al., Nature 590, 249 (2021) — alternating-twist trilayer
    graphene (Tc ~ 2.9 K).
  - Khalaf et al., PRB 100, 085109 (2019) — alternating-twist multilayer
    mapping: trilayer magic angle is sqrt(2) times the bilayer one.
  - Castro Neto et al., RMP 81, 109 (2009) — graphene band parameters.
  - Huder et al., PRL 120, 156405 (2018) — heterostrained twisted graphene
    layers (uniaxial strain of one layer reshapes the moire lattice).
  - Wang et al., Sci. Adv. 5, eaay8897 (2019) — composite super-moire
    lattices in double-aligned graphene heterostructures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waytogocoop.computation.moire import (
    _plane_wave_sum,
    _reciprocal_g_vectors_hexagonal,
    _reciprocal_g_vectors_square,
    moire_periodicity_1d,
    moire_periodicity_with_twist,
)
from waytogocoop.config import (
    BCS_GAP_RATIO,
    DEFAULT_GRID_SIZE,
    DELTA_TBG_MAX,
    DELTA_TTG_MAX,
    GRAPHENE_A,
    GRAPHENE_EXTENT_DEFAULT,
    HBAR_VF_GRAPHENE,
    KB_EV_K,
    NU_DOME_WIDTH,
    NU_OPTIMAL_FILLING,
    POISSON_GRAPHENE,
    SMALL_ANGLE_THRESHOLD,
    THETA_SC_WIDTH_DEG,
    W_INTERLAYER_TBG,
    ZERO_THRESHOLD,
)
from waytogocoop.materials.lattice import apply_rotation

STACKING_PRESETS: tuple[str, ...] = (
    "AA",
    "AB",
    "twisted_bilayer",
    "ABA",
    "ABC",
    "alternating_trilayer",
)

# Phase coefficients c_n for the repo's angle-ordered hexagonal G shell
# (angles 0, 60, ..., 300 deg).  With b1 at 0 deg and b2 at 60 deg the first
# shell in that angle order is {b1, b2, b2-b1, -b1, -b2, b1-b2}; the AB
# stacking offset tau = (a1 + a2)/3 gives phases G_n . tau = c_n * 2*pi/3
# with c_n = 1, 1, 0, -1, -1, 0.  Encoding registry offsets as phases keeps
# the potential grid-frame independent and rotation-safe.
_HEX_PHASE_COEFFS = (1, 1, 0, -1, -1, 0)


@dataclass
class LayerSpec:
    """One graphene layer within a stack."""

    twist_deg: float = 0.0        # CCW twist, same sign convention as apply_rotation
    stacking_index: int = 0       # 0 = A, 1 = B, 2 = C registry
    strain_percent: float = 0.0   # uniaxial heterostrain (percent elongation)
    strain_angle_deg: float = 0.0  # strain axis, measured from zigzag = x axis


def stack_layers(stacking: str, twist_angle_deg: float = 0.0) -> list[LayerSpec]:
    """Return the layer specification for a named stacking preset.

    Parameters
    ----------
    stacking : str
        One of ``STACKING_PRESETS``.
    twist_angle_deg : float
        Twist angle (degrees); only used by the ``"twisted_bilayer"`` and
        ``"alternating_trilayer"`` presets.

    Returns
    -------
    list[LayerSpec]
        One entry per layer, bottom to top.
    """
    presets: dict[str, list[LayerSpec]] = {
        "AA": [LayerSpec(0.0, 0), LayerSpec(0.0, 0)],
        "AB": [LayerSpec(0.0, 0), LayerSpec(0.0, 1)],
        "twisted_bilayer": [LayerSpec(0.0, 0), LayerSpec(twist_angle_deg, 0)],
        "ABA": [LayerSpec(0.0, 0), LayerSpec(0.0, 1), LayerSpec(0.0, 0)],
        "ABC": [LayerSpec(0.0, 0), LayerSpec(0.0, 1), LayerSpec(0.0, 2)],
        "alternating_trilayer": [
            LayerSpec(0.0, 0),
            LayerSpec(twist_angle_deg, 0),
            LayerSpec(0.0, 0),
        ],
    }
    if stacking not in presets:
        raise ValueError(
            f"Unknown stacking '{stacking}'. Available: {list(STACKING_PRESETS)}"
        )
    return presets[stacking]


def strained_g_vectors(
    g: np.ndarray,
    strain_percent: float,
    strain_angle_deg: float,
    poisson: float = POISSON_GRAPHENE,
) -> np.ndarray:
    """Apply a uniaxial heterostrain to a set of reciprocal lattice vectors.

    Small-strain linearization: real-space positions transform as
    ``r' = (I + E) r``, so reciprocal vectors transform as
    ``G' = (I + E)^{-T} G ~ (I - E) G`` to first order in the strain, with

    .. math::

        E = R(\\varphi)\\,
            \\mathrm{diag}(\\epsilon,\\; -\\nu\\epsilon)\\,
            R(\\varphi)^{T},
        \\qquad \\epsilon = \\text{strain\\_percent}/100

    (uniaxial elongation along the strain axis, Poisson contraction
    :math:`\\nu\\epsilon` transverse to it).  ``E`` is symmetric, so
    ``(I - E^T) = (I - E)``.

    Parameters
    ----------
    g : np.ndarray
        Shape ``(M, 2)`` reciprocal lattice vectors.
    strain_percent : float
        Uniaxial strain in percent (positive = elongation).
    strain_angle_deg : float
        Strain axis, measured from the zigzag = x axis (degrees).
    poisson : float
        Poisson ratio for the transverse contraction.

    Returns
    -------
    np.ndarray
        Strained vectors, same shape as *g*.
    """
    g = np.asarray(g, dtype=float)
    eps = strain_percent / 100.0
    phi = np.radians(strain_angle_deg)
    cos_p = np.cos(phi)
    sin_p = np.sin(phi)
    rot = np.array([[cos_p, -sin_p], [sin_p, cos_p]])
    strain = rot @ np.diag([eps, -poisson * eps]) @ rot.T
    return g @ (np.eye(2) - strain)


def _effective_g_shell(
    lattice_a: float,
    twist_deg: float = 0.0,
    strain_percent: float = 0.0,
    strain_angle_deg: float = 0.0,
) -> np.ndarray:
    """First hexagonal G shell with heterostrain applied before the twist."""
    g_vectors = _reciprocal_g_vectors_hexagonal(lattice_a)
    if abs(strain_percent) > ZERO_THRESHOLD:
        g_vectors = strained_g_vectors(g_vectors, strain_percent, strain_angle_deg)
    if abs(twist_deg) > ZERO_THRESHOLD:
        g_vectors = apply_rotation(g_vectors, twist_deg)
    return g_vectors


def layer_potential(
    x: np.ndarray,
    y: np.ndarray,
    lattice_a: float = GRAPHENE_A,
    twist_deg: float = 0.0,
    stacking_index: int = 0,
    strain_percent: float = 0.0,
    strain_angle_deg: float = 0.0,
    displacement: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate one layer's first-shell potential on a 2D grid.

    .. math::

        V(\\mathbf{r}) = \\sum_n \\cos(\\mathbf{G}_n \\cdot
            (\\mathbf{r} + \\mathbf{u}) - c_n \\phi)

    with registry phase :math:`\\phi = \\text{stacking\\_index} \\cdot 2\\pi/3`,
    coefficients :math:`c_n` from ``_HEX_PHASE_COEFFS``, and optional in-plane
    displacement field :math:`\\mathbf{u}` (curvature back-reaction).
    Heterostrain is applied to the G shell before the twist rotation.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays defining the grid axes (Angstrom).
    lattice_a : float
        In-plane lattice constant (Angstrom).
    twist_deg : float
        CCW twist applied to the G vectors (degrees).
    stacking_index : int
        Registry index: 0 = A, 1 = B, 2 = C.
    strain_percent : float
        Uniaxial heterostrain in percent (see ``strained_g_vectors``).
    strain_angle_deg : float
        Strain axis, measured from the zigzag = x axis (degrees).
    displacement : tuple[np.ndarray, np.ndarray] or None
        Optional ``(u_x, u_y)`` in-plane displacement arrays (Angstrom),
        each shaped like the ``(len(y), len(x))`` meshgrid.

    Returns
    -------
    np.ndarray
        Shape ``(len(y), len(x))`` potential field.
    """
    g_vectors = _effective_g_shell(lattice_a, twist_deg, strain_percent, strain_angle_deg)
    phi = stacking_index * 2.0 * np.pi / 3.0
    X, Y = np.meshgrid(x, y)
    if displacement is not None:
        u_x, u_y = displacement
        X = X + u_x
        Y = Y + u_y
    result = np.zeros_like(X, dtype=float)
    for gvec, coeff in zip(g_vectors, _HEX_PHASE_COEFFS, strict=True):
        result += np.cos(gvec[0] * X + gvec[1] * Y - coeff * phi)
    return result


def layer_potential_honeycomb(
    x: np.ndarray,
    y: np.ndarray,
    lattice_a: float = GRAPHENE_A,
    twist_deg: float = 0.0,
    stacking_index: int = 0,
    strain_percent: float = 0.0,
    strain_angle_deg: float = 0.0,
    displacement: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Evaluate one layer's potential with the honeycomb two-atom basis.

    .. math::

        V(\\mathbf{r}) = \\sum_n \\left[
            \\cos(\\mathbf{G}_n \\cdot (\\mathbf{r} - \\boldsymbol\\tau))
            + \\cos(\\mathbf{G}_n \\cdot (\\mathbf{r} - \\boldsymbol\\tau)
                    - c_n \\, 2\\pi/3) \\right]

    The second term is the B sublattice at
    :math:`\\boldsymbol\\delta = (\\mathbf{a}_1 + \\mathbf{a}_2)/3`, whose
    phases :math:`\\mathbf{G}_n \\cdot \\boldsymbol\\delta = c_n\\,2\\pi/3` use
    the same angle-ordered coefficients :math:`c_n` as the registry offset
    :math:`\\boldsymbol\\tau` (``_HEX_PHASE_COEFFS``).  At the origin with
    ``stacking_index=0``: ``6 + (4*cos(2*pi/3) + 2*cos(0)) = 6.0``.

    Parameters
    ----------
    x, y : np.ndarray
        1D arrays defining the grid axes (Angstrom).
    lattice_a : float
        In-plane lattice constant (Angstrom).
    twist_deg : float
        CCW twist applied to the G vectors (degrees).
    stacking_index : int
        Registry index: 0 = A, 1 = B, 2 = C.
    strain_percent : float
        Uniaxial heterostrain in percent (see ``strained_g_vectors``).
    strain_angle_deg : float
        Strain axis, measured from the zigzag = x axis (degrees).
    displacement : tuple[np.ndarray, np.ndarray] or None
        Optional ``(u_x, u_y)`` in-plane displacement arrays (Angstrom),
        each shaped like the ``(len(y), len(x))`` meshgrid.

    Returns
    -------
    np.ndarray
        Shape ``(len(y), len(x))`` potential field.
    """
    g_vectors = _effective_g_shell(lattice_a, twist_deg, strain_percent, strain_angle_deg)
    phi = stacking_index * 2.0 * np.pi / 3.0
    X, Y = np.meshgrid(x, y)
    if displacement is not None:
        u_x, u_y = displacement
        X = X + u_x
        Y = Y + u_y
    result = np.zeros_like(X, dtype=float)
    for gvec, coeff in zip(g_vectors, _HEX_PHASE_COEFFS, strict=True):
        phase = gvec[0] * X + gvec[1] * Y - coeff * phi
        result += np.cos(phase) + np.cos(phase - coeff * (2.0 * np.pi / 3.0))
    return result


def moire_period_from_g_shells(g_sub: np.ndarray, g_over: np.ndarray) -> float:
    """Moire period from two angle-order-paired hexagonal first G shells.

    .. math::

        L = \\frac{4\\pi}{\\sqrt{3}\\,
            \\min_i |\\mathbf{G}^{sub}_i - \\mathbf{G}^{over}_i|}

    Since the first-shell magnitude is :math:`|G| = 4\\pi/(\\sqrt{3}a)`, this
    generalizes both the mismatch formula ``a1*a2/|a1 - a2|`` and the twist
    formula ``a/(2*sin(theta/2))`` (twist-only shells give
    :math:`|\\Delta G| = 2|G|\\sin(\\theta/2)` exactly) to arbitrary
    strained/twisted shells.

    Parameters
    ----------
    g_sub, g_over : np.ndarray
        Shape ``(M, 2)`` shells in the same angle order (e.g. from
        ``strained_g_vectors`` / ``apply_rotation`` on the same base shell).

    Returns
    -------
    float
        Moire period in Angstrom; ``inf`` when the shells coincide
        (``min |dG| < ZERO_THRESHOLD``).
    """
    g_sub = np.asarray(g_sub, dtype=float)
    g_over = np.asarray(g_over, dtype=float)
    if g_sub.shape != g_over.shape:
        raise ValueError(
            f"G shells must have the same shape, got {g_sub.shape} and {g_over.shape}"
        )
    dg_min = float(np.linalg.norm(g_sub - g_over, axis=1).min())
    if dg_min < ZERO_THRESHOLD:
        return float(np.inf)
    return float(4.0 * np.pi / (np.sqrt(3.0) * dg_min))


def generate_stack_pattern(
    stacking: str = "twisted_bilayer",
    twist_angle_deg: float = 1.08,
    lattice_a: float = GRAPHENE_A,
    grid_size: int = DEFAULT_GRID_SIZE,
    physical_extent: float = GRAPHENE_EXTENT_DEFAULT,
) -> dict:
    """Generate the interference pattern for a graphene stack.

    The pattern is the product of the per-layer potentials, normalized to
    [0, 1] exactly like ``generate_moire_pattern``.

    Parameters
    ----------
    stacking : str
        One of ``STACKING_PRESETS``.
    twist_angle_deg : float
        Twist angle (degrees) for the twisted presets.
    lattice_a : float
        In-plane lattice constant (Angstrom).
    grid_size : int
        Number of grid points per axis.
    physical_extent : float
        Half-width of the real-space window (Angstrom).

    Returns
    -------
    dict
        ``x`` (1D ndarray), ``y`` (1D ndarray), ``pattern`` (2D ndarray shape
        ``(grid_size, grid_size)``), ``moire_period`` (float, Angstrom; inf
        for untwisted stacks), ``n_layers`` (int).

    Notes
    -----
    Untwisted stacks (AA, AB, ABA, ABC) have the atomic-scale period
    a = 2.46 Angstrom, so callers must choose ``physical_extent`` small
    enough that ``dx = 2*physical_extent/(grid_size - 1)`` is at most about
    a/3 to resolve the pattern (UI presets use ~15 Angstrom for these).

    Delegates to ``generate_stack_pattern_v2`` with the v2 features
    (honeycomb basis, heterostrain, displacement) disabled; the result is
    bit-identical to the v1 implementation.
    """
    return generate_stack_pattern_v2(
        stacking=stacking,
        twist_angle_deg=twist_angle_deg,
        lattice_a=lattice_a,
        grid_size=grid_size,
        physical_extent=physical_extent,
    )


def generate_stack_pattern_v2(
    stacking: str = "twisted_bilayer",
    twist_angle_deg: float = 1.08,
    lattice_a: float = GRAPHENE_A,
    grid_size: int = DEFAULT_GRID_SIZE,
    physical_extent: float = GRAPHENE_EXTENT_DEFAULT,
    honeycomb: bool = False,
    heterostrain_percent: float = 0.0,
    heterostrain_angle_deg: float = 0.0,
    displacement: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Generate a graphene-stack pattern with honeycomb basis, heterostrain, and warp.

    Like ``generate_stack_pattern`` (product of per-layer potentials,
    normalized to [0, 1]) with three v2 extensions:

    * ``honeycomb=True`` uses the two-atom-basis potential
      ``layer_potential_honeycomb`` for every layer.
    * Heterostrain is applied to layer index 1 only — experimentally, strain
      relaxation leaves one layer pinned to the substrate while the
      transferred layer carries the uniaxial strain (Huder et al., PRL 120,
      156405 (2018)).
    * ``displacement`` (curvature back-reaction) is applied to ALL layers —
      a rigid bending of the whole sheet stack.

    Parameters
    ----------
    stacking : str
        One of ``STACKING_PRESETS``.
    twist_angle_deg : float
        Twist angle (degrees) for the twisted presets.
    lattice_a : float
        In-plane lattice constant (Angstrom).
    grid_size : int
        Number of grid points per axis.
    physical_extent : float
        Half-width of the real-space window (Angstrom).
    honeycomb : bool
        Use the honeycomb two-atom basis potential.
    heterostrain_percent : float
        Uniaxial heterostrain (percent) applied to layer index 1.
    heterostrain_angle_deg : float
        Strain axis, measured from the zigzag = x axis (degrees).
    displacement : tuple[np.ndarray, np.ndarray] or None
        Optional ``(u_x, u_y)`` in-plane displacement arrays (Angstrom),
        each of shape ``(grid_size, grid_size)``.

    Returns
    -------
    dict
        ``x`` (1D ndarray), ``y`` (1D ndarray), ``pattern`` (2D ndarray shape
        ``(grid_size, grid_size)``), ``moire_period`` (float, Angstrom;
        computed from the layer-0/layer-1 effective G shells when strained,
        otherwise from the twist formula — inf for untwisted, unstrained
        stacks), ``n_layers`` (int).
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if physical_extent <= 0:
        raise ValueError("physical_extent must be positive")
    if lattice_a <= 0:
        raise ValueError("lattice_a must be positive")
    layers = stack_layers(stacking, twist_angle_deg)
    if abs(heterostrain_percent) > ZERO_THRESHOLD:
        layers[1].strain_percent = heterostrain_percent
        layers[1].strain_angle_deg = heterostrain_angle_deg
    if displacement is not None:
        u_x, u_y = displacement
        u_x = np.asarray(u_x, dtype=float)
        u_y = np.asarray(u_y, dtype=float)
        expected = (grid_size, grid_size)
        if u_x.shape != expected or u_y.shape != expected:
            raise ValueError(
                f"displacement arrays must have shape {expected}, "
                f"got {u_x.shape} and {u_y.shape}"
            )
        displacement = (u_x, u_y)

    x = np.linspace(-physical_extent, physical_extent, grid_size)
    y = np.linspace(-physical_extent, physical_extent, grid_size)

    potential = layer_potential_honeycomb if honeycomb else layer_potential
    pattern = np.ones((grid_size, grid_size))
    for layer in layers:
        pattern = pattern * potential(
            x,
            y,
            lattice_a,
            layer.twist_deg,
            layer.stacking_index,
            strain_percent=layer.strain_percent,
            strain_angle_deg=layer.strain_angle_deg,
            displacement=displacement,
        )
    pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
    p_min = pattern.min()
    p_max = pattern.max()
    if p_max - p_min > ZERO_THRESHOLD:
        pattern = (pattern - p_min) / (p_max - p_min)
    else:
        pattern = np.zeros_like(pattern)

    if abs(heterostrain_percent) > ZERO_THRESHOLD:
        moire_period = moire_period_from_g_shells(
            _effective_g_shell(
                lattice_a,
                layers[0].twist_deg,
                layers[0].strain_percent,
                layers[0].strain_angle_deg,
            ),
            _effective_g_shell(
                lattice_a,
                layers[1].twist_deg,
                layers[1].strain_percent,
                layers[1].strain_angle_deg,
            ),
        )
    else:
        twists = [layer.twist_deg for layer in layers]
        theta_rel = max(twists) - min(twists)
        if abs(theta_rel) > SMALL_ANGLE_THRESHOLD:
            moire_period = moire_periodicity_with_twist(lattice_a, theta_rel)
        else:
            moire_period = np.inf

    return {
        "x": x,
        "y": y,
        "pattern": pattern,
        "moire_period": moire_period,
        "n_layers": len(layers),
    }


def generate_supermoire_pattern(
    stacking: str = "twisted_bilayer",
    twist_angle_deg: float = 1.08,
    overlayer_a: float = 4.264,
    overlayer_lattice_type: str = "hexagonal",
    interface_twist_deg: float = 0.0,
    lattice_a: float = GRAPHENE_A,
    grid_size: int = DEFAULT_GRID_SIZE,
    physical_extent: float = GRAPHENE_EXTENT_DEFAULT,
    honeycomb: bool = False,
) -> dict:
    """Supermoire: a graphene stack on an additional (substrate) overlayer.

    The pattern is the product of the graphene-stack layer potentials and the
    overlayer plane-wave potential (twisted by ``interface_twist_deg``),
    min-max normalized to [0, 1].  Two moire wavelengths coexist — the
    intra-stack one and the graphene/overlayer interface one — and their beat
    is the supermoire period (Wang et al., Sci. Adv. 5, eaay8897 (2019)).

    Parameters
    ----------
    stacking : str
        One of ``STACKING_PRESETS``.
    twist_angle_deg : float
        Intra-stack twist angle (degrees) for the twisted presets.
    overlayer_a : float
        Overlayer lattice constant (Angstrom); default is Sb2Te3.
    overlayer_lattice_type : str
        ``"hexagonal"`` or ``"square"``.
    interface_twist_deg : float
        Twist between the graphene layer 0 and the overlayer (degrees).
    lattice_a : float
        Graphene in-plane lattice constant (Angstrom).
    grid_size : int
        Number of grid points per axis.
    physical_extent : float
        Half-width of the real-space window (Angstrom).
    honeycomb : bool
        Use the honeycomb two-atom basis for the graphene layers.

    Returns
    -------
    dict
        ``x``, ``y`` (1D ndarrays), ``pattern`` (2D ndarray shape
        ``(grid_size, grid_size)``), ``stack_period`` (float, Angstrom —
        intra-stack moire period, as in ``generate_stack_pattern_v2``),
        ``interface_period`` (float — graphene/overlayer moire period),
        ``supermoire_period`` (float — beat of the two periods,
        ``L1*L2/|L1 - L2|``; inf when either period is inf or they
        coincide), ``n_layers`` (int — graphene layers plus the overlayer).

    Notes
    -----
    For a hexagonal overlayer the interface period comes from
    ``moire_period_from_g_shells`` (mismatch and twist combined).  A square
    overlayer shell cannot be angle-order paired with the hexagonal graphene
    shell, so the 1D mismatch formula ``a1*a2/|a1 - a2|`` is used instead —
    an approximation that ignores ``interface_twist_deg``.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if physical_extent <= 0:
        raise ValueError("physical_extent must be positive")
    if lattice_a <= 0:
        raise ValueError("lattice_a must be positive")
    if overlayer_a <= 0:
        raise ValueError("overlayer_a must be positive")
    if overlayer_lattice_type not in ("hexagonal", "square"):
        raise ValueError(
            f"overlayer_lattice_type must be 'hexagonal' or 'square', "
            f"got '{overlayer_lattice_type}'"
        )
    layers = stack_layers(stacking, twist_angle_deg)

    x = np.linspace(-physical_extent, physical_extent, grid_size)
    y = np.linspace(-physical_extent, physical_extent, grid_size)

    potential = layer_potential_honeycomb if honeycomb else layer_potential
    pattern = np.ones((grid_size, grid_size))
    for layer in layers:
        pattern = pattern * potential(
            x, y, lattice_a, layer.twist_deg, layer.stacking_index
        )

    if overlayer_lattice_type == "hexagonal":
        g_over = _reciprocal_g_vectors_hexagonal(overlayer_a)
    else:
        g_over = _reciprocal_g_vectors_square(overlayer_a)
    if abs(interface_twist_deg) > ZERO_THRESHOLD:
        g_over = apply_rotation(g_over, interface_twist_deg)
    pattern = pattern * _plane_wave_sum(g_over, x, y)

    pattern = np.nan_to_num(pattern, nan=0.0, posinf=0.0, neginf=0.0)
    p_min = pattern.min()
    p_max = pattern.max()
    if p_max - p_min > ZERO_THRESHOLD:
        pattern = (pattern - p_min) / (p_max - p_min)
    else:
        pattern = np.zeros_like(pattern)

    twists = [layer.twist_deg for layer in layers]
    theta_rel = max(twists) - min(twists)
    if abs(theta_rel) > SMALL_ANGLE_THRESHOLD:
        stack_period = moire_periodicity_with_twist(lattice_a, theta_rel)
    else:
        stack_period = np.inf

    if overlayer_lattice_type == "hexagonal":
        interface_period = moire_period_from_g_shells(
            _effective_g_shell(lattice_a, layers[0].twist_deg), g_over
        )
    else:
        interface_period = moire_periodicity_1d(lattice_a, overlayer_a)

    if np.isfinite(stack_period) and np.isfinite(interface_period):
        supermoire_period = moire_periodicity_1d(stack_period, interface_period)
    else:
        supermoire_period = np.inf

    return {
        "x": x,
        "y": y,
        "pattern": pattern,
        "stack_period": stack_period,
        "interface_period": interface_period,
        "supermoire_period": float(supermoire_period),
        "n_layers": len(layers) + 1,
    }


def magic_angle_deg(
    n_layers: int = 2,
    w_ev: float = W_INTERLAYER_TBG,
    hbar_vf: float = HBAR_VF_GRAPHENE,
    lattice_a: float = GRAPHENE_A,
) -> float:
    """First magic angle of twisted bilayer / alternating-twist trilayer graphene.

    First-order Bistritzer-MacDonald condition :math:`\\alpha = 1/\\sqrt{3}`:

    .. math::

        \\theta_m = 2 \\arcsin\\!\\left(
            \\frac{\\sqrt{3}\\, w_{\\text{eff}}}{2 \\hbar v_F k_D}\\right),
        \\qquad k_D = \\frac{4\\pi}{3a}

    with :math:`w_{\\text{eff}} = \\sqrt{2}\\, w` for the alternating-twist
    trilayer (Khalaf-Vishwanath mapping) and :math:`w_{\\text{eff}} = w` for
    the bilayer.

    Parameters
    ----------
    n_layers : int
        2 (twisted bilayer) or 3 (alternating-twist trilayer).
    w_ev : float
        Interlayer tunneling amplitude (eV).
    hbar_vf : float
        Monolayer Dirac velocity hbar*v_F (eV*Angstrom).
    lattice_a : float
        In-plane lattice constant (Angstrom).

    Returns
    -------
    float
        Magic angle in degrees: 1.076 for n_layers=2, 1.521 for n_layers=3.
    """
    if n_layers not in (2, 3):
        raise ValueError(f"n_layers must be 2 or 3, got {n_layers}")
    k_dirac = 4.0 * np.pi / (3.0 * lattice_a)
    w_eff = w_ev * np.sqrt(2.0) if n_layers == 3 else w_ev
    arg = np.sqrt(3.0) * w_eff / (2.0 * hbar_vf * k_dirac)
    if arg >= 1.0:
        raise ValueError(
            f"No magic angle: sqrt(3)*w/(2*hbar_vf*k_D) = {arg:.3g} >= 1"
        )
    return float(np.degrees(2.0 * np.arcsin(arg)))


def dirac_velocity_ratio(
    twist_angle_deg,
    w_ev: float = W_INTERLAYER_TBG,
    hbar_vf: float = HBAR_VF_GRAPHENE,
    lattice_a: float = GRAPHENE_A,
) -> float | np.ndarray:
    """Renormalized Dirac velocity of twisted bilayer graphene, v*/v_F.

    First-order Bistritzer-MacDonald result:

    .. math::

        \\frac{v^*}{v_F} = \\frac{1 - 3\\alpha^2}{1 + 6\\alpha^2},
        \\qquad \\alpha = \\frac{w}{\\hbar v_F k_\\theta},
        \\qquad k_\\theta = 2 k_D \\sin(\\theta/2)

    Exactly zero at the magic angle.  Valid for small twists
    (theta <~ 3 deg); at larger angles it saturates toward 1.

    Parameters
    ----------
    twist_angle_deg : float or array_like
        Twist angle in degrees (must be positive).
    w_ev : float
        Interlayer tunneling amplitude (eV).
    hbar_vf : float
        Monolayer Dirac velocity hbar*v_F (eV*Angstrom).
    lattice_a : float
        In-plane lattice constant (Angstrom).

    Returns
    -------
    float or np.ndarray
        Velocity ratio v*/v_F (negative past the magic angle in this
        first-order expression).
    """
    theta = np.asarray(twist_angle_deg, dtype=float)
    if np.any(theta <= 0):
        raise ValueError("twist_angle_deg must be positive")
    k_dirac = 4.0 * np.pi / (3.0 * lattice_a)
    k_theta = 2.0 * k_dirac * np.sin(np.radians(theta) / 2.0)
    alpha = w_ev / (hbar_vf * k_theta)
    result = (1.0 - 3.0 * alpha**2) / (1.0 + 6.0 * alpha**2)
    return float(result) if result.ndim == 0 else result


def filling_dome_factor(
    filling: float,
    nu_opt: float = NU_OPTIMAL_FILLING,
    nu_width: float = NU_DOME_WIDTH,
) -> float:
    """SPECULATIVE: superconducting dome factor versus moire-band filling.

    Parabolic dome centred at the optimal filling:

    .. math::

        D(\\nu) = \\max\\!\\left(0,\\;
            1 - \\left(\\frac{|\\nu| - \\nu_{\\text{opt}}}
                            {\\nu_w}\\right)^2\\right)

    Parameters
    ----------
    filling : float
        Electrons per moire cell, nu; the flat band holds |nu| <= 4.
    nu_opt : float
        Filling at the dome maximum.
    nu_width : float
        Dome half-width in filling.

    Returns
    -------
    float
        Dimensionless factor in [0, 1].
    """
    if abs(filling) > 4:
        raise ValueError(f"|filling| must be <= 4 electrons per moire cell, got {filling}")
    return max(0.0, 1.0 - ((abs(filling) - nu_opt) / nu_width) ** 2)


@dataclass
class FlatBandSC:
    """Flat-band superconductivity estimates for a twisted graphene stack."""

    alpha: float             # dimensionless interlayer coupling w_eff/(hbar*v_F*k_theta)
    velocity_ratio: float    # v*/v_F, first-order Bistritzer-MacDonald
    theta_magic_deg: float   # first magic angle for this stack (degrees)
    delta_mev: float         # SPECULATIVE gap estimate (meV)
    tc_kelvin: float         # SPECULATIVE critical temperature (K)
    dome_factor: float       # SPECULATIVE filling-dome factor in [0, 1]


def compute_flat_band_sc(
    twist_angle_deg: float,
    n_layers: int = 2,
    filling: float = NU_OPTIMAL_FILLING,
    w_ev: float = W_INTERLAYER_TBG,
    hbar_vf: float = HBAR_VF_GRAPHENE,
    lattice_a: float = GRAPHENE_A,
) -> FlatBandSC:
    """Estimate flat-band superconductivity for a twisted graphene stack.

    The band-structure quantities (alpha, velocity ratio, magic angle) are
    first-order Bistritzer-MacDonald / Khalaf-Vishwanath results.  The gap
    model is SPECULATIVE — a Lorentzian in twist angle times a filling dome:

    .. math::

        \\Delta(\\theta, \\nu) = \\Delta_{\\max} \\,
            \\frac{1}{1 + \\left(\\frac{\\theta - \\theta_m}
                                      {\\Gamma_\\theta}\\right)^2} \\, D(\\nu),
        \\qquad T_c = \\frac{\\Delta}{1.764\\, k_B}

    with :math:`\\Delta_{\\max}` = ``DELTA_TBG_MAX`` (bilayer) or
    ``DELTA_TTG_MAX`` (alternating-twist trilayer).

    Parameters
    ----------
    twist_angle_deg : float
        Twist angle in degrees (must be positive).
    n_layers : int
        2 (twisted bilayer) or 3 (alternating-twist trilayer).
    filling : float
        Electrons per moire cell, |filling| <= 4.
    w_ev : float
        Interlayer tunneling amplitude (eV); scaled by sqrt(2) internally
        for n_layers = 3.
    hbar_vf : float
        Monolayer Dirac velocity hbar*v_F (eV*Angstrom).
    lattice_a : float
        In-plane lattice constant (Angstrom).

    Returns
    -------
    FlatBandSC
        Band and superconductivity estimates.

    Notes
    -----
    The CPDM contrast of any moire-modulated gap is tiny here:
    ``cpdm_amplitude(130.5, XI_TBG)`` ~ 0.022 because the coherence length
    (xi ~ 500 Angstrom) far exceeds the moire period — the modulation washes
    out, which is the expected physical behavior.
    """
    if twist_angle_deg <= 0:
        raise ValueError("twist_angle_deg must be positive")
    if n_layers not in (2, 3):
        raise ValueError(f"n_layers must be 2 or 3, got {n_layers}")

    w_eff = w_ev * np.sqrt(2.0) if n_layers == 3 else w_ev
    theta_magic = magic_angle_deg(n_layers, w_ev, hbar_vf, lattice_a)
    k_dirac = 4.0 * np.pi / (3.0 * lattice_a)
    k_theta = 2.0 * k_dirac * np.sin(np.radians(twist_angle_deg) / 2.0)
    alpha = w_eff / (hbar_vf * k_theta)
    velocity_ratio = dirac_velocity_ratio(twist_angle_deg, w_eff, hbar_vf, lattice_a)

    dome = filling_dome_factor(filling)
    delta_max = DELTA_TBG_MAX if n_layers == 2 else DELTA_TTG_MAX
    lorentzian = 1.0 / (1.0 + ((twist_angle_deg - theta_magic) / THETA_SC_WIDTH_DEG) ** 2)
    delta_mev = delta_max * lorentzian * dome
    tc_kelvin = delta_mev * 1.0e-3 / (BCS_GAP_RATIO * KB_EV_K)

    return FlatBandSC(
        alpha=alpha,
        velocity_ratio=velocity_ratio,
        theta_magic_deg=theta_magic,
        delta_mev=delta_mev,
        tc_kelvin=tc_kelvin,
        dome_factor=dome,
    )
