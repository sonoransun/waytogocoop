//! Bistritzer-MacDonald continuum model for twisted bilayer graphene:
//! moire band structure, flat-band width and density of states.
//!
//! Established physics unless marked **SPECULATIVE**.
//!
//! Literature basis for the continuum (plane-wave) moire band model:
//!   - Bistritzer & MacDonald, PNAS 108, 12233 (2011) — continuum model,
//!     magic-angle condition
//!   - Koshino et al., PRX 8, 031087 (2018) — corrugation-corrected
//!     tunnelling amplitudes w_AA < w_AB
//!   - Tarnopolsky, Kruchkov & Vishwanath, PRL 122, 106405 (2019) — chiral
//!     limit (w_AA = 0) with exact particle-hole symmetric flat bands
//!
//! Model conventions (valley xi = +1 or -1):
//!   k_D = 4*pi/(3a), k_theta = 2*k_D*sin(theta/2);
//!   q1 = k_theta*(0,-1), q2 = k_theta*(sqrt3/2,1/2), q3 = k_theta*(-sqrt3/2,1/2);
//!   moire reciprocal basis b1 = q2 - q1, b2 = q3 - q1.
//!   Momentum lattice: layer-1 sites {m*b1 + n*b2}, layer-2 sites
//!   {q1 + m*b1 + n*b2}, truncated to |m|, |n|, |m+n| <= n_shells; two
//!   sublattices per site.  Diagonal Dirac blocks
//!   h(k - Q) = hbar_vf*(xi*kx*sigma_x + ky*sigma_y); off-diagonal blocks
//!   T_j = w_aa*I + w_ab*(cos(2pi(j-1)/3)*sigma_x + xi*sin(2pi(j-1)/3)*sigma_y)
//!   couple layer-1 site Q to layer-2 site Q + q_j.
//!
//! Eigensolver: nalgebra's self-adjoint path (`symmetric_eigenvalues`) is
//! used directly on the complex Hermitian matrix — it Householder-
//! tridiagonalizes and rephases the off-diagonal to a real symmetric
//! tridiagonal problem, which is exact for Hermitian input (validated
//! against a known spectrum in the tests).
//!
//! Mirrors `src/waytogocoop/computation/bm_model.py` in the Python stack —
//! changes here must land in both.

use nalgebra::{Complex, DMatrix};
use rayon::prelude::*;
use std::collections::HashMap;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// AA-region interlayer tunnelling amplitude in eV (Koshino 2018).
pub const W_AA_TBG: f64 = 0.0797;
/// AB-region interlayer tunnelling amplitude in eV (Koshino 2018).
pub const W_AB_TBG: f64 = 0.0975;

// ---------------------------------------------------------------------------
// Configuration and results
// ---------------------------------------------------------------------------

/// Configuration for a Bistritzer-MacDonald band computation.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BMConfig {
    /// Twist angle in degrees (must be positive).
    pub twist_angle_deg: f64,
    /// AA-region interlayer tunnelling in eV.
    pub w_aa: f64,
    /// AB-region interlayer tunnelling in eV.
    pub w_ab: f64,
    /// Dirac velocity as hbar*v_F in eV*Angstrom.
    pub hbar_vf: f64,
    /// Graphene lattice constant in Angstrom.
    pub lattice_a: f64,
    /// Momentum-lattice truncation: |m|, |n|, |m+n| <= n_shells.
    pub n_shells: usize,
    /// Valley index xi: +1 for K, -1 for K'.
    pub valley: i32,
}

impl Default for BMConfig {
    fn default() -> Self {
        Self {
            twist_angle_deg: 1.08,
            w_aa: W_AA_TBG,
            w_ab: W_AB_TBG,
            hbar_vf: 5.96,
            lattice_a: 2.46,
            n_shells: 3,
            valley: 1,
        }
    }
}

/// Moire band structure along the K -> Gamma -> M -> K' path.
#[derive(Debug, Clone)]
pub struct BandStructure {
    /// Cumulative distance along the k-path in 1/Angstrom, one per k-point.
    pub k_distances: Vec<f64>,
    /// Sorted (ascending) eigenvalues in meV, one Vec per k-point.
    pub energies_mev: Vec<Vec<f64>>,
    /// Path distances of the K, Gamma, M, K' corners.
    pub tick_positions: Vec<f64>,
    /// High-symmetry point labels matching `tick_positions`.
    pub tick_labels: Vec<&'static str>,
    /// Total energy span of the two central (flat) bands over the path, meV.
    pub flat_bandwidth_mev: f64,
    /// Smaller of the gaps separating the flat bands from the remote bands
    /// above and below, in meV; 0 when they overlap in energy.
    pub flat_gap_mev: f64,
}

/// Density of states from a uniform moire-BZ sampling.
#[derive(Debug, Clone)]
pub struct DosResult {
    /// Histogram bin centers in meV.
    pub energies_mev: Vec<f64>,
    /// Gaussian-broadened density of states per meV per k-point (>= 0).
    pub dos: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Momentum lattice
// ---------------------------------------------------------------------------

/// Truncated momentum lattice of Dirac-cone replicas for both layers.
struct MomentumLattice {
    /// Site momenta in 1/Angstrom, all layer-1 sites first.
    sites: Vec<[f64; 2]>,
    /// Interlayer bonds (layer-1 index, layer-2 index, j in 0..3) with
    /// coupling matrix T_{j+1}.
    bonds: Vec<(usize, usize, usize)>,
    /// Moire wavevector magnitude k_theta in 1/Angstrom.
    k_theta: f64,
    /// Moire reciprocal basis b1, b2 in 1/Angstrom.
    b1: [f64; 2],
    b2: [f64; 2],
}

fn validate_config(cfg: &BMConfig) -> Result<(), String> {
    if cfg.twist_angle_deg <= 0.0 {
        return Err(format!(
            "twist_angle_deg must be positive, got {}",
            cfg.twist_angle_deg
        ));
    }
    if cfg.w_aa < 0.0 {
        return Err(format!("w_aa must be non-negative, got {}", cfg.w_aa));
    }
    if cfg.w_ab < 0.0 {
        return Err(format!("w_ab must be non-negative, got {}", cfg.w_ab));
    }
    if cfg.hbar_vf <= 0.0 {
        return Err(format!("hbar_vf must be positive, got {}", cfg.hbar_vf));
    }
    if cfg.lattice_a <= 0.0 {
        return Err(format!("lattice_a must be positive, got {}", cfg.lattice_a));
    }
    if cfg.n_shells < 1 {
        return Err(format!("n_shells must be >= 1, got {}", cfg.n_shells));
    }
    if cfg.valley != 1 && cfg.valley != -1 {
        return Err(format!("valley must be +1 or -1, got {}", cfg.valley));
    }
    Ok(())
}

fn momentum_lattice(cfg: &BMConfig) -> MomentumLattice {
    let k_d = 4.0 * PI / (3.0 * cfg.lattice_a);
    let k_theta = 2.0 * k_d * (cfg.twist_angle_deg.to_radians() / 2.0).sin();
    let sq3 = 3.0_f64.sqrt();
    let q1 = [0.0, -k_theta];
    let q2 = [k_theta * sq3 / 2.0, k_theta / 2.0];
    let q3 = [-k_theta * sq3 / 2.0, k_theta / 2.0];
    let b1 = [q2[0] - q1[0], q2[1] - q1[1]];
    let b2 = [q3[0] - q1[0], q3[1] - q1[1]];

    let ns = cfg.n_shells as i32;
    let mut mn: Vec<(i32, i32)> = Vec::new();
    let mut index_of: HashMap<(i32, i32), usize> = HashMap::new();
    for m in -ns..=ns {
        for n in -ns..=ns {
            if (m + n).abs() <= ns {
                index_of.insert((m, n), mn.len());
                mn.push((m, n));
            }
        }
    }

    let n_per_layer = mn.len();
    let mut sites = Vec::with_capacity(2 * n_per_layer);
    for &(m, n) in &mn {
        let (m, n) = (m as f64, n as f64);
        sites.push([m * b1[0] + n * b2[0], m * b1[1] + n * b2[1]]);
    }
    for &(m, n) in &mn {
        let (m, n) = (m as f64, n as f64);
        sites.push([q1[0] + m * b1[0] + n * b2[0], q1[1] + m * b1[1] + n * b2[1]]);
    }

    // Layer-1 site Q couples to layer-2 sites Q + q_j; in lattice indices
    // q1 -> (m, n), q2 = q1 + b1 -> (m+1, n), q3 = q1 + b2 -> (m, n+1).
    let mut bonds = Vec::new();
    for (i1, &(m, n)) in mn.iter().enumerate() {
        for (j, (dm, dn)) in [(0, 0), (1, 0), (0, 1)].into_iter().enumerate() {
            if let Some(&i2) = index_of.get(&(m + dm, n + dn)) {
                bonds.push((i1, n_per_layer + i2, j));
            }
        }
    }

    MomentumLattice {
        sites,
        bonds,
        k_theta,
        b1,
        b2,
    }
}

// ---------------------------------------------------------------------------
// Hamiltonian assembly and diagonalization
// ---------------------------------------------------------------------------

/// Assemble the BM Hamiltonian at Bloch momentum k (units of 1/Angstrom);
/// entries in eV.  Dimension is 2 * sites (two sublattices per site).
fn build_hamiltonian(cfg: &BMConfig, lat: &MomentumLattice, k: [f64; 2]) -> DMatrix<Complex<f64>> {
    let dim = 2 * lat.sites.len();
    let xi = cfg.valley as f64;
    let mut h = DMatrix::<Complex<f64>>::zeros(dim, dim);

    // Diagonal Dirac blocks h(k + Q) = hbar_vf*(xi*px*sigma_x + py*sigma_y);
    // k + Q (not k - Q) so the layer-2 cone sits at k = -q1, matching the
    // Python mirror and the K' path corner.
    for (s, site) in lat.sites.iter().enumerate() {
        let px = k[0] + site[0];
        let py = k[1] + site[1];
        let off = Complex::new(cfg.hbar_vf * xi * px, -cfg.hbar_vf * py);
        h[(2 * s, 2 * s + 1)] = off;
        h[(2 * s + 1, 2 * s)] = off.conj();
    }

    // Tunnelling blocks T_{j+1} = w_aa*I + w_ab*(cos(2pi*j/3)*sigma_x
    // + xi*sin(2pi*j/3)*sigma_y), Hermitian 2x2.
    let t_blocks: [[[Complex<f64>; 2]; 2]; 3] = std::array::from_fn(|j| {
        let phi = 2.0 * PI * j as f64 / 3.0;
        let diag = Complex::new(cfg.w_aa, 0.0);
        let od = Complex::new(cfg.w_ab * phi.cos(), -xi * cfg.w_ab * phi.sin());
        [[diag, od], [od.conj(), diag]]
    });

    for &(i1, i2, j) in &lat.bonds {
        let t = &t_blocks[j];
        for (r, row) in t.iter().enumerate() {
            for (c, val) in row.iter().enumerate() {
                h[(2 * i2 + r, 2 * i1 + c)] = *val;
                h[(2 * i1 + c, 2 * i2 + r)] = val.conj();
            }
        }
    }

    h
}

/// Eigenvalues (ascending, eV) of a complex Hermitian matrix via nalgebra's
/// self-adjoint `symmetric_eigenvalues` path.
fn hermitian_eigenvalues_sorted(h: &DMatrix<Complex<f64>>) -> Vec<f64> {
    let mut evals: Vec<f64> = h.symmetric_eigenvalues().iter().cloned().collect();
    evals.sort_by(f64::total_cmp);
    evals
}

/// Sorted eigenvalues in meV at Bloch momentum k.
fn eigenvalues_mev(cfg: &BMConfig, lat: &MomentumLattice, k: [f64; 2]) -> Vec<f64> {
    let h = build_hamiltonian(cfg, lat, k);
    hermitian_eigenvalues_sorted(&h)
        .into_iter()
        .map(|e| e * 1e3)
        .collect()
}

// ---------------------------------------------------------------------------
// Band structure along K -> Gamma -> M -> K'
// ---------------------------------------------------------------------------

/// Build the K -> Gamma -> M -> K' path with `n_per_segment` steps per
/// segment.  Returns (k-points, cumulative distances, corner distances).
///
/// K = (0, 0) is the layer-1 Dirac point and K' = -q1 = (0, +k_theta) the
/// layer-2 Dirac point (the q1 offset of the layer-2 momentum lattice puts
/// its cone at k = -q1).  Gamma is the centre of the moire BZ hexagon whose
/// adjacent corners are those two cones, M the midpoint of the edge between
/// them.  Mirrors `_high_symmetry_points` in computation/bm_model.py.
fn k_path(k_theta: f64, n_per_segment: usize) -> (Vec<[f64; 2]>, Vec<f64>, Vec<f64>) {
    let sq3 = 3.0_f64.sqrt();
    let corners = [
        [0.0, 0.0],                              // K
        [k_theta * sq3 / 2.0, k_theta / 2.0],    // Gamma
        [0.0, k_theta / 2.0],                    // M
        [0.0, k_theta],                          // K'
    ];

    let mut k_points = vec![corners[0]];
    let mut distances = vec![0.0];
    let mut ticks = vec![0.0];
    for seg in 0..3 {
        let (start, end) = (corners[seg], corners[seg + 1]);
        for i in 1..=n_per_segment {
            let t = i as f64 / n_per_segment as f64;
            let p = [
                start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]),
            ];
            let prev = *k_points.last().unwrap();
            let step = ((p[0] - prev[0]).powi(2) + (p[1] - prev[1]).powi(2)).sqrt();
            distances.push(distances.last().unwrap() + step);
            k_points.push(p);
        }
        ticks.push(*distances.last().unwrap());
    }
    (k_points, distances, ticks)
}

/// Compute the moire band structure along K -> Gamma -> M -> K'.
pub fn compute_band_structure(
    cfg: &BMConfig,
    n_k_per_segment: usize,
) -> Result<BandStructure, String> {
    validate_config(cfg)?;
    if n_k_per_segment < 1 {
        return Err(format!(
            "n_k_per_segment must be >= 1, got {}",
            n_k_per_segment
        ));
    }

    let lat = momentum_lattice(cfg);
    let (k_points, k_distances, tick_positions) = k_path(lat.k_theta, n_k_per_segment);

    let energies_mev: Vec<Vec<f64>> = k_points
        .par_iter()
        .map(|&k| eigenvalues_mev(cfg, &lat, k))
        .collect();

    // Flat bands = the two bands closest to zero energy (indices dim/2 - 1
    // and dim/2 of the ascending spectrum at every k).
    let dim = 2 * lat.sites.len();
    let (lo, hi) = (dim / 2 - 1, dim / 2);
    let fold_max = |band: usize| {
        energies_mev
            .iter()
            .map(|e| e[band])
            .fold(f64::MIN, f64::max)
    };
    let fold_min = |band: usize| {
        energies_mev
            .iter()
            .map(|e| e[band])
            .fold(f64::MAX, f64::min)
    };
    let flat_bandwidth_mev = fold_max(hi) - fold_min(lo);
    let gap_above = fold_min(hi + 1) - fold_max(hi);
    let gap_below = fold_min(lo) - fold_max(lo - 1);
    let flat_gap_mev = gap_above.min(gap_below).max(0.0);

    Ok(BandStructure {
        k_distances,
        energies_mev,
        tick_positions,
        tick_labels: vec!["K", "Γ", "M", "K'"],
        flat_bandwidth_mev,
        flat_gap_mev,
    })
}

/// Flat-band width in meV over the K -> Gamma -> M -> K' path.
pub fn flat_band_width_mev(cfg: &BMConfig, n_k_per_segment: usize) -> Result<f64, String> {
    Ok(compute_band_structure(cfg, n_k_per_segment)?.flat_bandwidth_mev)
}

// ---------------------------------------------------------------------------
// Density of states
// ---------------------------------------------------------------------------

/// Gaussian-broadened density of states from an n_k_grid x n_k_grid uniform
/// sampling of the moire Brillouin zone, restricted to |E| <= e_window_mev.
pub fn compute_dos(
    cfg: &BMConfig,
    n_k_grid: usize,
    e_window_mev: f64,
    n_bins: usize,
    broadening_mev: f64,
) -> Result<DosResult, String> {
    validate_config(cfg)?;
    if n_k_grid < 1 {
        return Err(format!("n_k_grid must be >= 1, got {}", n_k_grid));
    }
    if e_window_mev <= 0.0 {
        return Err(format!(
            "e_window_mev must be positive, got {}",
            e_window_mev
        ));
    }
    if n_bins < 1 {
        return Err(format!("n_bins must be >= 1, got {}", n_bins));
    }
    if broadening_mev <= 0.0 {
        return Err(format!(
            "broadening_mev must be positive, got {}",
            broadening_mev
        ));
    }

    let lat = momentum_lattice(cfg);
    let (b1, b2) = (lat.b1, lat.b2);
    let cutoff = e_window_mev + 5.0 * broadening_mev;

    let eigenvalues: Vec<f64> = (0..n_k_grid * n_k_grid)
        .into_par_iter()
        .flat_map_iter(|idx| {
            let fx = (idx / n_k_grid) as f64 / n_k_grid as f64;
            let fy = (idx % n_k_grid) as f64 / n_k_grid as f64;
            let k = [fx * b1[0] + fy * b2[0], fx * b1[1] + fy * b2[1]];
            eigenvalues_mev(cfg, &lat, k)
                .into_iter()
                .filter(move |e| e.abs() <= cutoff)
        })
        .collect();

    let bin_width = 2.0 * e_window_mev / n_bins as f64;
    let energies_mev: Vec<f64> = (0..n_bins)
        .map(|i| -e_window_mev + (i as f64 + 0.5) * bin_width)
        .collect();
    let norm = 1.0
        / (broadening_mev * (2.0 * PI).sqrt() * (n_k_grid * n_k_grid) as f64);
    let dos: Vec<f64> = energies_mev
        .iter()
        .map(|&e| {
            eigenvalues
                .iter()
                .map(|&ev| {
                    let z = (e - ev) / broadening_mev;
                    (-0.5 * z * z).exp()
                })
                .sum::<f64>()
                * norm
        })
        .collect();

    Ok(DosResult { energies_mev, dos })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphene::dirac_velocity_ratio;

    #[test]
    fn test_eigensolver_known_hermitian() {
        // H = [[2, i, 0], [-i, 2, 0], [0, 0, 3]] has eigenvalues {1, 3, 3}.
        let i = Complex::new(0.0, 1.0);
        let two = Complex::new(2.0, 0.0);
        let three = Complex::new(3.0, 0.0);
        let zero = Complex::new(0.0, 0.0);
        let h = DMatrix::from_row_slice(3, 3, &[two, i, zero, -i, two, zero, zero, zero, three]);
        let evals = hermitian_eigenvalues_sorted(&h);
        let expected = [1.0, 3.0, 3.0];
        for (e, x) in evals.iter().zip(expected.iter()) {
            assert!((e - x).abs() < 1e-10, "evals = {:?}", evals);
        }
    }

    #[test]
    fn test_hamiltonian_hermitian() {
        let cfg = BMConfig::default();
        let lat = momentum_lattice(&cfg);
        let k = [0.37 * lat.b1[0] + 0.21 * lat.b2[0], 0.37 * lat.b1[1] + 0.21 * lat.b2[1]];
        let h = build_hamiltonian(&cfg, &lat, k);
        let max_dev = (h.adjoint() - &h)
            .iter()
            .map(|c| c.norm())
            .fold(0.0_f64, f64::max);
        assert!(max_dev < 1e-12, "max deviation = {}", max_dev);
    }

    #[test]
    fn test_dirac_degeneracy_at_k_point() {
        let cfg = BMConfig::default();
        let lat = momentum_lattice(&cfg);
        let energies = eigenvalues_mev(&cfg, &lat, [0.0, 0.0]);
        let min_abs = energies.iter().map(|e| e.abs()).fold(f64::MAX, f64::min);
        assert!(min_abs < 5.0, "min |E| = {} meV", min_abs);
    }

    #[test]
    fn test_layer2_dirac_cone_at_k_prime_tick() {
        // K' = -q1 = (0, +k_theta) carries the layer-2 Dirac cone; the mirror
        // point (0, -k_theta) is not a Dirac point in this convention. Pins
        // the path-label orientation (mirrors the Python test).
        let cfg = BMConfig::default();
        let k_d = 4.0 * PI / (3.0 * cfg.lattice_a);
        let k_theta = 2.0 * k_d * (cfg.twist_angle_deg.to_radians() / 2.0).sin();
        let lat = momentum_lattice(&cfg);
        let min_abs = |k: [f64; 2]| {
            eigenvalues_mev(&cfg, &lat, k)
                .iter()
                .map(|e| e.abs())
                .fold(f64::MAX, f64::min)
        };
        let at_k_prime = min_abs([0.0, k_theta]);
        let at_mirror = min_abs([0.0, -k_theta]);
        assert!(at_k_prime < 5.0, "min |E| at K' = {} meV", at_k_prime);
        assert!(
            at_mirror > 2.0 * at_k_prime,
            "mirror point should not host a cone: {} vs {}",
            at_mirror,
            at_k_prime
        );
    }

    #[test]
    fn test_chiral_limit_particle_hole_symmetry() {
        let cfg = BMConfig {
            w_aa: 0.0,
            ..Default::default()
        };
        let lat = momentum_lattice(&cfg);
        // Gamma point of the moire BZ
        let k = [lat.k_theta * 3.0_f64.sqrt() / 2.0, -lat.k_theta / 2.0];
        let energies = eigenvalues_mev(&cfg, &lat, k);
        let n = energies.len();
        for i in 0..n {
            let dev = (energies[i] + energies[n - 1 - i]).abs();
            assert!(dev < 1e-6, "E[{}] = {}, E[{}] = {}", i, energies[i], n - 1 - i, energies[n - 1 - i]);
        }
    }

    #[test]
    fn test_velocity_matches_perturbative_ratio() {
        // At 2.5 deg with w_aa = w_ab = 0.110 the numerical slope at K should
        // match the analytic (1 - 3a^2)/(1 + 6a^2) ratio used in graphene.rs.
        let cfg = BMConfig {
            twist_angle_deg: 2.5,
            w_aa: 0.110,
            w_ab: 0.110,
            ..Default::default()
        };
        let lat = momentum_lattice(&cfg);
        let dk = 0.02 * lat.k_theta;
        let energies = eigenvalues_mev(&cfg, &lat, [dk, 0.0]);
        let e_plus_ev = energies[energies.len() / 2] * 1e-3;
        let v_num = e_plus_ev / (cfg.hbar_vf * dk);
        let v_analytic = dirac_velocity_ratio(2.5);
        let rel = (v_num / v_analytic - 1.0).abs();
        assert!(rel < 0.12, "v_num = {}, v_analytic = {}", v_num, v_analytic);
    }

    #[test]
    fn test_magic_angle_bandwidth_minimum() {
        let width_at = |theta: f64| {
            flat_band_width_mev(
                &BMConfig {
                    twist_angle_deg: theta,
                    ..Default::default()
                },
                6,
            )
            .unwrap()
        };
        let angles: Vec<f64> = (0..=8).map(|i| 0.9 + 0.05 * i as f64).collect();
        let widths: Vec<f64> = angles.iter().map(|&t| width_at(t)).collect();
        let (i_min, w_min) = widths
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, w)| (i, *w))
            .unwrap();
        assert!(
            i_min > 0 && i_min < widths.len() - 1,
            "minimum at scan edge: theta = {}, widths = {:?}",
            angles[i_min],
            widths
        );
        assert!(w_min < 25.0, "W_min = {} meV", w_min);
        let w_large = width_at(2.0);
        assert!(
            w_large > 4.0 * w_min,
            "W(2.0) = {}, W_min = {}",
            w_large,
            w_min
        );
    }

    #[test]
    fn test_valley_spectra_agree_at_k_point() {
        let cfg_plus = BMConfig::default();
        let cfg_minus = BMConfig {
            valley: -1,
            ..Default::default()
        };
        let lat = momentum_lattice(&cfg_plus);
        let e_plus = eigenvalues_mev(&cfg_plus, &lat, [0.0, 0.0]);
        let e_minus = eigenvalues_mev(&cfg_minus, &lat, [0.0, 0.0]);
        for (a, b) in e_plus.iter().zip(e_minus.iter()) {
            assert!((a - b).abs() < 1e-6, "E+ = {}, E- = {}", a, b);
        }
    }

    #[test]
    fn test_band_structure_shape_and_ticks() {
        let cfg = BMConfig::default();
        let bands = compute_band_structure(&cfg, 4).unwrap();
        assert_eq!(bands.k_distances.len(), 3 * 4 + 1);
        assert_eq!(bands.energies_mev.len(), bands.k_distances.len());
        assert_eq!(bands.tick_positions.len(), 4);
        assert_eq!(bands.tick_labels, vec!["K", "Γ", "M", "K'"]);
        assert!((bands.tick_positions[0] - 0.0).abs() < 1e-15);
        assert!(bands
            .tick_positions
            .windows(2)
            .all(|w| w[1] > w[0]));
        assert!((bands.tick_positions[3] - bands.k_distances.last().unwrap()).abs() < 1e-12);
        assert!(bands.flat_bandwidth_mev >= 0.0);
        assert!(bands.flat_gap_mev >= 0.0);
        // Every per-k spectrum is sorted ascending.
        for e in &bands.energies_mev {
            assert!(e.windows(2).all(|w| w[1] >= w[0]));
        }
    }

    #[test]
    fn test_dos_nonnegative_with_flat_band_peak() {
        let cfg = BMConfig::default();
        let dos = compute_dos(&cfg, 6, 50.0, 64, 3.0).unwrap();
        assert_eq!(dos.energies_mev.len(), 64);
        assert_eq!(dos.dos.len(), 64);
        assert!(dos.dos.iter().all(|&v| v >= 0.0 && v.is_finite()));
        let (i_max, _) = dos
            .dos
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        let e_peak = dos.energies_mev[i_max];
        assert!(e_peak.abs() < 15.0, "DOS peak at {} meV", e_peak);
    }

    #[test]
    fn test_invalid_inputs() {
        let base = BMConfig::default();
        let bad_configs = [
            BMConfig { twist_angle_deg: 0.0, ..base },
            BMConfig { twist_angle_deg: -1.0, ..base },
            BMConfig { w_aa: -0.1, ..base },
            BMConfig { w_ab: -0.1, ..base },
            BMConfig { hbar_vf: 0.0, ..base },
            BMConfig { lattice_a: 0.0, ..base },
            BMConfig { n_shells: 0, ..base },
            BMConfig { valley: 0, ..base },
        ];
        for cfg in &bad_configs {
            assert!(compute_band_structure(cfg, 4).is_err(), "cfg = {:?}", cfg);
            assert!(compute_dos(cfg, 4, 50.0, 32, 3.0).is_err(), "cfg = {:?}", cfg);
        }
        assert!(compute_band_structure(&base, 0).is_err());
        assert!(compute_dos(&base, 0, 50.0, 32, 3.0).is_err());
        assert!(compute_dos(&base, 4, 0.0, 32, 3.0).is_err());
        assert!(compute_dos(&base, 4, 50.0, 0, 3.0).is_err());
        assert!(compute_dos(&base, 4, 50.0, 32, 0.0).is_err());
    }
}
