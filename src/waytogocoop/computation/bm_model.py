"""Bistritzer-MacDonald continuum model for twisted bilayer graphene: moire band
structure and density of states.

Established physics unless marked SPECULATIVE.

The Hamiltonian is built on a momentum lattice around one valley of the moire
Brillouin zone: layer-1 plane-wave sites at {m*b1 + n*b2}, layer-2 sites at
{q1 + m*b1 + n*b2}, two sublattices per site.  Sites are kept when the
hexagonal-shell indices satisfy |m|, |n|, |m + n| <= n_shells (a standard
hexagonal truncation of the momentum lattice; n_shells = 3 converges the flat
bands to well below the meV scale probed here).  First-order model: the
+/- theta/2 rotation of the Pauli matrices in the layer Dirac blocks is
neglected, as in the original Bistritzer-MacDonald treatment.

Energies are handled internally in eV (``build_bm_hamiltonian`` returns eV);
band structures and densities of states are exposed in meV.

Mirrors ``crates/moire-core/src/bm_model.rs``.

Literature basis:
  - Bistritzer & MacDonald, PNAS 108, 12233 (2011) — continuum model of
    twisted bilayer graphene, first magic angle.
  - Koshino et al., PRX 8, 031087 (2018) — corrugation-corrected tunneling
    amplitudes w_aa < w_ab.
  - Tarnopolsky, Kruchkov & Vishwanath, PRL 122, 106405 (2019) — chiral
    limit (w_aa = 0) with exact particle-hole symmetric flat bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from waytogocoop.config import (
    BM_KPOINTS_DEFAULT,
    BM_SHELLS_DEFAULT,
    DOS_BROADENING_MEV,
    GRAPHENE_A,
    HBAR_VF_GRAPHENE,
    W_AA_TBG,
    W_AB_TBG,
)

EV_TO_MEV: float = 1.0e3

_SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)


@dataclass
class BMConfig:
    """Configuration of the Bistritzer-MacDonald continuum Hamiltonian."""

    twist_angle_deg: float = 1.08     # relative twist (degrees), > 0
    w_aa: float = W_AA_TBG            # eV — AA interlayer tunneling, >= 0
    w_ab: float = W_AB_TBG            # eV — AB interlayer tunneling, > 0
    hbar_vf: float = HBAR_VF_GRAPHENE  # eV*Angstrom — monolayer Dirac velocity
    lattice_a: float = GRAPHENE_A     # Angstrom — graphene lattice constant
    n_shells: int = BM_SHELLS_DEFAULT  # momentum-lattice truncation, >= 1
    valley: int = 1                   # +1 (K) or -1 (K')


@dataclass
class BandStructure:
    """Moire band structure along the K -> Gamma -> M -> K' path."""

    k_distances: np.ndarray      # (n_k,) cumulative path length (1/Angstrom)
    energies_mev: np.ndarray     # (n_k, n_bands) eigenvalues, sorted per k (meV)
    tick_positions: list[float]  # path distances of the high-symmetry points
    tick_labels: list[str]       # ['K', 'Γ', 'M', "K'"]
    flat_bandwidth_mev: float    # max - min over the two middle bands along the path
    flat_gap_mev: float          # min band-edge gap from the flat bands to the next
                                 # bands above/below (negative if they overlap)


def _validate_config(config: BMConfig) -> None:
    if config.twist_angle_deg <= 0:
        raise ValueError("twist_angle_deg must be positive")
    if config.n_shells < 1:
        raise ValueError("n_shells must be >= 1")
    if config.valley not in (1, -1):
        raise ValueError(f"valley must be +1 or -1, got {config.valley}")
    if config.w_aa < 0:
        raise ValueError("w_aa must be >= 0")
    if config.w_ab <= 0:
        raise ValueError("w_ab must be positive")


def _k_theta(config: BMConfig) -> float:
    """Moire wavevector k_theta = 2*k_D*sin(theta/2) with k_D = 4*pi/(3*a)."""
    k_dirac = 4.0 * np.pi / (3.0 * config.lattice_a)
    return 2.0 * k_dirac * np.sin(np.radians(config.twist_angle_deg) / 2.0)


def _momentum_transfers(k_theta: float) -> np.ndarray:
    """The three interlayer momentum transfers q1, q2, q3, shape (3, 2)."""
    return k_theta * np.array([
        [0.0, -1.0],
        [np.sqrt(3.0) / 2.0, 0.5],
        [-np.sqrt(3.0) / 2.0, 0.5],
    ])


def _site_indices(n_shells: int) -> list[tuple[int, int]]:
    """Hexagonally truncated (m, n) site indices: |m|, |n|, |m+n| <= n_shells."""
    return [
        (m, n)
        for m in range(-n_shells, n_shells + 1)
        for n in range(-n_shells, n_shells + 1)
        if abs(m + n) <= n_shells
    ]


def build_bm_hamiltonian(kx: float, ky: float, config: BMConfig) -> np.ndarray:
    """Build the BM continuum Hamiltonian at Bloch momentum (kx, ky).

    The momentum is measured from the layer-1 Dirac point (1/Angstrom);
    layer-2 momentum-lattice sites already carry the q1 offset.  Diagonal
    Dirac blocks are ``hbar_vf * (xi*kx_loc*sigma_x + ky_loc*sigma_y)`` with
    ``k_loc = k + Q_site``; layer-1 site Q couples to layer-2 site Q + q_j via

    .. math::

        T_j = w_{aa}\\,\\sigma_0 + w_{ab}\\left[
            \\cos\\tfrac{2\\pi(j-1)}{3}\\,\\sigma_x
            + \\xi \\sin\\tfrac{2\\pi(j-1)}{3}\\,\\sigma_y\\right],
        \\qquad j = 1, 2, 3.

    Parameters
    ----------
    kx, ky : float
        Bloch momentum relative to the layer-1 Dirac point (1/Angstrom).
    config : BMConfig
        Model parameters (validated here).

    Returns
    -------
    np.ndarray
        Complex Hermitian matrix of shape ``(dim, dim)`` in eV, where
        ``dim = 2 * (n_sites_layer1 + n_sites_layer2)``.
    """
    _validate_config(config)
    xi = config.valley
    q = _momentum_transfers(_k_theta(config))
    b1 = q[1] - q[0]
    b2 = q[2] - q[0]
    sites = _site_indices(config.n_shells)
    index_of = {site: i for i, site in enumerate(sites)}
    n_sites = len(sites)
    dim = 4 * n_sites
    hamiltonian = np.zeros((dim, dim), dtype=complex)

    for layer_offset, q_layer in ((0, np.zeros(2)), (n_sites, q[0])):
        for i, (m, n) in enumerate(sites):
            q_site = q_layer + m * b1 + n * b2
            block = config.hbar_vf * (
                xi * (kx + q_site[0]) * _SIGMA_X + (ky + q_site[1]) * _SIGMA_Y
            )
            row = 2 * (layer_offset + i)
            hamiltonian[row:row + 2, row:row + 2] = block

    # q_j = q1 + {0, b1, b2}: layer-2 partner of layer-1 site (m, n) is
    # (m, n) + delta_j in the shared (m, n) indexing.
    deltas = ((0, 0), (1, 0), (0, 1))
    for j, (dm, dn) in enumerate(deltas):
        angle = 2.0 * np.pi * j / 3.0
        t_j = config.w_aa * np.eye(2, dtype=complex) + config.w_ab * (
            np.cos(angle) * _SIGMA_X + xi * np.sin(angle) * _SIGMA_Y
        )
        for i, (m, n) in enumerate(sites):
            partner = index_of.get((m + dm, n + dn))
            if partner is None:
                continue
            row = 2 * i
            col = 2 * (n_sites + partner)
            hamiltonian[row:row + 2, col:col + 2] = t_j
            hamiltonian[col:col + 2, row:row + 2] = t_j.conj().T
    return hamiltonian


def _high_symmetry_points(k_theta: float) -> list[np.ndarray]:
    """Path corners K_m, Gamma_m, M_m, K'_m in the layer-1-Dirac frame.

    K_m = (0, 0) is the layer-1 Dirac point and K'_m = -q1 = (0, +k_theta)
    the layer-2 Dirac point (the site q1 offsets the layer-2 momentum
    lattice, so its cone sits at k = -q1).  Gamma_m is the centre of the
    moire BZ hexagon whose adjacent corners are those two cones, and M_m
    the midpoint of the edge between them.
    """
    return [
        np.array([0.0, 0.0]),
        k_theta * np.array([np.sqrt(3.0) / 2.0, 0.5]),
        np.array([0.0, k_theta / 2.0]),
        np.array([0.0, k_theta]),
    ]


def compute_band_structure(
    config: BMConfig, n_k_per_segment: int = BM_KPOINTS_DEFAULT
) -> BandStructure:
    """Diagonalize the BM Hamiltonian along the K -> Gamma -> M -> K' path.

    Parameters
    ----------
    config : BMConfig
        Model parameters.
    n_k_per_segment : int
        Number of k-points added per path segment (total ``3*n + 1`` points
        including the shared corners).

    Returns
    -------
    BandStructure
        Sorted eigenvalues in meV per k-point, path metadata, and the flat-band
        width/gap diagnostics (flat bands = the two middle bands).
    """
    _validate_config(config)
    if n_k_per_segment < 1:
        raise ValueError("n_k_per_segment must be >= 1")
    corners = _high_symmetry_points(_k_theta(config))
    k_points = [corners[0]]
    for start, end in zip(corners[:-1], corners[1:], strict=True):
        for step in range(1, n_k_per_segment + 1):
            k_points.append(start + (end - start) * step / n_k_per_segment)
    k_array = np.array(k_points)

    k_distances = np.zeros(len(k_array))
    k_distances[1:] = np.cumsum(np.linalg.norm(np.diff(k_array, axis=0), axis=1))

    energies_mev = EV_TO_MEV * np.array([
        np.linalg.eigvalsh(build_bm_hamiltonian(kx, ky, config)) for kx, ky in k_array
    ])

    mid = energies_mev.shape[1] // 2
    flat = energies_mev[:, mid - 1:mid + 1]
    flat_bandwidth_mev = float(flat.max() - flat.min())
    gap_above = float(energies_mev[:, mid + 1].min() - energies_mev[:, mid].max())
    gap_below = float(energies_mev[:, mid - 1].min() - energies_mev[:, mid - 2].max())

    tick_positions = [float(k_distances[i * n_k_per_segment]) for i in range(4)]
    return BandStructure(
        k_distances=k_distances,
        energies_mev=energies_mev,
        tick_positions=tick_positions,
        tick_labels=["K", "Γ", "M", "K'"],
        flat_bandwidth_mev=flat_bandwidth_mev,
        flat_gap_mev=min(gap_above, gap_below),
    )


def flat_band_width_mev(config: BMConfig, n_k_per_segment: int = 8) -> float:
    """Convenience: flat-band width (meV) along the high-symmetry path."""
    return compute_band_structure(config, n_k_per_segment).flat_bandwidth_mev


def compute_dos(
    config: BMConfig,
    n_k_grid: int = 12,
    e_window_mev: float = 150.0,
    n_bins: int = 200,
    broadening_mev: float = DOS_BROADENING_MEV,
) -> dict:
    """Gaussian-broadened density of states over the moire Brillouin zone.

    Eigenvalues are collected on a uniform ``n_k_grid x n_k_grid`` grid of
    fractional offsets over the parallelogram spanned by the moire reciprocal
    basis b1, b2 and histogrammed with Gaussian broadening.  The DOS is
    normalized per k-point, so it integrates to the average number of states
    per k-point inside the energy window.

    Parameters
    ----------
    config : BMConfig
        Model parameters.
    n_k_grid : int
        Number of k-points per reciprocal basis direction.
    e_window_mev : float
        Half-width of the energy window around zero (meV).
    n_bins : int
        Number of energy bins.
    broadening_mev : float
        Gaussian broadening sigma (meV).

    Returns
    -------
    dict
        ``energies_mev`` (1D ndarray, ``(n_bins,)`` bin centers) and ``dos``
        (1D ndarray, ``(n_bins,)`` states per meV per k-point).
    """
    _validate_config(config)
    if n_k_grid < 1:
        raise ValueError("n_k_grid must be >= 1")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2")
    if e_window_mev <= 0:
        raise ValueError("e_window_mev must be positive")
    if broadening_mev <= 0:
        raise ValueError("broadening_mev must be positive")

    q = _momentum_transfers(_k_theta(config))
    b1 = q[1] - q[0]
    b2 = q[2] - q[0]

    energies_mev = np.linspace(-e_window_mev, e_window_mev, n_bins)
    dos = np.zeros(n_bins)
    for i in range(n_k_grid):
        for j in range(n_k_grid):
            k = (i / n_k_grid) * b1 + (j / n_k_grid) * b2
            eigenvalues = EV_TO_MEV * np.linalg.eigvalsh(
                build_bm_hamiltonian(k[0], k[1], config)
            )
            kept = eigenvalues[np.abs(eigenvalues) <= e_window_mev]
            if kept.size:
                dos += np.exp(
                    -((energies_mev[:, None] - kept[None, :]) ** 2)
                    / (2.0 * broadening_mev**2)
                ).sum(axis=1)
    dos /= n_k_grid * n_k_grid * np.sqrt(2.0 * np.pi) * broadening_mev
    return {"energies_mev": energies_mev, "dos": dos}
