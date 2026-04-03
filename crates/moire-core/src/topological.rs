//! Topological surface state physics and 3D Cooper surface construction.
//!
//! Implements proximity effect z-decay, 3D gap field extension, Dirac surface
//! state dispersion, and speculative features (Majorana zero modes, topological
//! phase boundaries, magnetoelectric response, Chern number estimates).
//!
//! Established physics unless marked **SPECULATIVE**.

use rayon::prelude::*;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Reduced Planck constant in eV·s.
const HBAR_EV_S: f64 = 6.582119569e-16;
/// Default TI surface Dirac velocity in m/s.
#[cfg(test)]
const DEFAULT_VF: f64 = 5.0e5;
/// Default Majorana localization length in Angstrom.
#[cfg(test)]
const DEFAULT_XI_M: f64 = 50.0;
/// Default TI surface Fermi wavevector in 1/Angstrom.
#[cfg(test)]
const DEFAULT_KF: f64 = 0.1;
/// Bohr magneton in eV/T.
const MU_B_EV_T: f64 = 5.788e-5;
/// Elementary charge in Coulombs.
const E_CHARGE: f64 = 1.602176634e-19;
/// Planck constant in J·s.
const H_PLANCK: f64 = 6.62607015e-34;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Configuration for proximity effect z-decay.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProximityConfig {
    /// Proximity coherence length in Angstrom.
    pub xi_prox: f64,
    /// Number of z-slices.
    pub n_z_layers: usize,
    /// Minimum z (into SC), Angstrom.
    pub z_min: f64,
    /// Maximum z (into TI), Angstrom.
    pub z_max: f64,
    /// Interface transmission coefficient T ∈ (0, 1].
    pub interface_transparency: f64,
}

impl Default for ProximityConfig {
    fn default() -> Self {
        Self {
            xi_prox: 100.0,
            n_z_layers: 50,
            z_min: -50.0,
            z_max: 300.0,
            interface_transparency: 0.8,
        }
    }
}

/// 3D gap field with z-decay.
#[derive(Debug, Clone)]
pub struct ProximityResult {
    /// 3D gap field: row-major z * ny * nx, in meV.
    pub gap_3d: Vec<f64>,
    /// z-coordinates in Angstrom.
    pub z_coords: Vec<f64>,
    /// 1D decay profile f(z).
    pub decay_profile: Vec<f64>,
    /// xy grid resolution.
    pub resolution: usize,
    /// Number of z layers.
    pub n_z: usize,
}

/// **SPECULATIVE**: Majorana zero mode probability density.
#[derive(Debug, Clone)]
pub struct MajoranaResult {
    /// 2D probability density, row-major.
    pub probability_density: Vec<f64>,
    /// Majorana localization length in Angstrom.
    pub localization_length: f64,
    /// Number of vortex cores hosting MZMs.
    pub n_vortices_with_mzm: usize,
    /// Grid resolution.
    pub resolution: usize,
}

/// Full 3D Cooper-pair order parameter surface.
#[derive(Debug, Clone)]
pub struct CooperSurface3D {
    /// 3D field: z * ny * nx, row-major.
    pub field_3d: Vec<f64>,
    /// x-coordinates (Angstrom).
    pub x_coords: Vec<f64>,
    /// y-coordinates (Angstrom).
    pub y_coords: Vec<f64>,
    /// z-coordinates (Angstrom).
    pub z_coords: Vec<f64>,
    /// Moire period (Angstrom).
    pub moire_period: f64,
    /// Vortex lattice period (Angstrom, inf if no field).
    pub vortex_period: f64,
    /// xy grid resolution.
    pub resolution: usize,
}

// ---------------------------------------------------------------------------
// Established physics — proximity effect
// ---------------------------------------------------------------------------

/// Proximity decay profile f(z).
///
/// z < 0 (inside SC): f = 1.0
/// z ≥ 0 (interface + TI): f = T * exp(-z / xi_prox)
pub fn proximity_decay_profile(
    z_coords: &[f64],
    xi_prox: f64,
    interface_transparency: f64,
) -> Vec<f64> {
    z_coords
        .iter()
        .map(|&z| {
            if z < 0.0 {
                1.0
            } else {
                interface_transparency * (-z / xi_prox).exp()
            }
        })
        .collect()
}

/// Extend 2D gap field to 3D via proximity z-decay.
///
/// gap_2d is row-major, resolution x resolution.
pub fn compute_gap_3d(
    gap_2d: &[f64],
    resolution: usize,
    config: &ProximityConfig,
) -> ProximityResult {
    let n_z = config.n_z_layers;
    let n_xy = resolution * resolution;

    let z_coords: Vec<f64> = (0..n_z)
        .map(|i| config.z_min + (config.z_max - config.z_min) * i as f64 / (n_z - 1).max(1) as f64)
        .collect();

    let profile = proximity_decay_profile(&z_coords, config.xi_prox, config.interface_transparency);

    let mut gap_3d = Vec::with_capacity(n_z * n_xy);
    for iz in 0..n_z {
        let f = profile[iz];
        for ixy in 0..n_xy {
            gap_3d.push(gap_2d[ixy] * f);
        }
    }

    ProximityResult {
        gap_3d,
        z_coords,
        decay_profile: profile,
        resolution,
        n_z,
    }
}

// ---------------------------------------------------------------------------
// Established physics — Dirac surface state
// ---------------------------------------------------------------------------

/// Dirac cone dispersion: E(k) = hbar * v_F * |k|.
///
/// k in 1/Angstrom, returns meV.
pub fn dirac_dispersion(k: f64, v_f: f64) -> f64 {
    // k in 1/Ang = 1e10 /m
    HBAR_EV_S * v_f * k.abs() * 1e10 * 1e3 // eV → meV
}

// ---------------------------------------------------------------------------
// 3D Cooper surface construction
// ---------------------------------------------------------------------------

/// Construct full 3D Cooper-pair order parameter surface.
pub fn compute_cooper_surface_3d(
    gap_2d: &[f64],
    resolution: usize,
    physical_extent: f64,
    moire_period: f64,
    vortex_period: f64,
    suppression: Option<&[f64]>,
    proximity_config: &ProximityConfig,
) -> CooperSurface3D {
    // Apply suppression
    let effective_gap: Vec<f64> = match suppression {
        Some(s) => gap_2d.iter().zip(s.iter()).map(|(&g, &s)| g * s).collect(),
        None => gap_2d.to_vec(),
    };

    let result = compute_gap_3d(&effective_gap, resolution, proximity_config);

    // Build coordinate arrays
    let n = resolution;
    let x_coords: Vec<f64> = (0..n)
        .map(|i| (i as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent)
        .collect();
    let y_coords = x_coords.clone();

    CooperSurface3D {
        field_3d: result.gap_3d,
        x_coords,
        y_coords,
        z_coords: result.z_coords,
        moire_period,
        vortex_period,
        resolution,
    }
}

// ---------------------------------------------------------------------------
// SPECULATIVE — Majorana zero modes
// ---------------------------------------------------------------------------

/// Bessel function J_0(x) — polynomial approximation (Abramowitz & Stegun).
///
/// Accurate to ~1e-6 for |x| < 100.
fn bessel_j0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7
            + y * (-11214424.18 + y * (77392.33017 + y * (-184.9052456)))));
        let den = 57568490411.0 + y * (1029532985.0 + y * (9494680.718
            + y * (59272.64853 + y * (267.8532712 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 0.785398164;
        let p = 1.0 + y * (-0.1098628627e-2 + y * (0.2734510407e-4
            + y * (-0.2073370639e-5 + y * 0.2093887211e-6)));
        let q = -0.1562499995e-1 + y * (0.1430488765e-3
            + y * (-0.6911147651e-5 + y * (0.7621095161e-6 - y * 0.934935152e-7)));
        (0.636619772 / ax).sqrt() * (xx.cos() * p - z * xx.sin() * q)
    }
}

/// **SPECULATIVE**: Majorana zero mode probability density.
///
/// |psi(r)|^2 ~ sum_v exp(-2|r-r_v|/xi_M) * J_0(k_F|r-r_v|)^2
///
/// Fu & Kane, PRL 100, 096407 (2008).
pub fn majorana_probability_density(
    resolution: usize,
    physical_extent: f64,
    vortex_positions: &[[f64; 2]],
    xi_m: f64,
    k_f: f64,
) -> MajoranaResult {
    let n = resolution;

    if vortex_positions.is_empty() {
        return MajoranaResult {
            probability_density: vec![0.0_f64; n * n],
            localization_length: xi_m,
            n_vortices_with_mzm: 0,
            resolution,
        };
    }

    let mut density: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map(|iy| {
            let y = (iy as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
            (0..n).map(move |ix| {
                let x = (ix as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
                let mut val = 0.0_f64;
                for &[vx, vy] in vortex_positions {
                    let r = ((x - vx).powi(2) + (y - vy).powi(2)).sqrt();
                    let envelope = (-2.0 * r / xi_m).exp();
                    let osc = bessel_j0(k_f * r).powi(2);
                    val += envelope * osc;
                }
                val
            }).collect::<Vec<f64>>()
        })
        .collect();

    // Normalise
    let peak = density.iter().copied().fold(0.0_f64, f64::max);
    if peak > 1e-30 {
        for v in density.iter_mut() {
            *v /= peak;
        }
    }

    MajoranaResult {
        probability_density: density,
        localization_length: xi_m,
        n_vortices_with_mzm: vortex_positions.len(),
        resolution,
    }
}

// ---------------------------------------------------------------------------
// SPECULATIVE — topological phase boundary
// ---------------------------------------------------------------------------

/// **SPECULATIVE**: Topological phase index (Fu-Kane criterion).
///
/// Topological (1) when E_Z > sqrt(Delta^2 + mu^2), trivial (0) otherwise.
pub fn topological_phase_index(zeeman_mev: f64, delta_mev: f64, mu_mev: f64) -> i32 {
    let threshold = (delta_mev * delta_mev + mu_mev * mu_mev).sqrt();
    if zeeman_mev > threshold { 1 } else { 0 }
}

/// **SPECULATIVE**: Phase diagram over (B, Delta) space.
///
/// Returns row-major array: delta_idx * n_b + b_idx.
pub fn phase_diagram_sweep(
    b_values: &[f64],
    delta_values: &[f64],
    g_factor: f64,
    mu_mev: f64,
) -> Vec<i32> {
    let n_b = b_values.len();
    let n_d = delta_values.len();
    let mut phase = vec![0_i32; n_d * n_b];

    for (i, &delta) in delta_values.iter().enumerate() {
        for (j, &b) in b_values.iter().enumerate() {
            let e_z = g_factor * MU_B_EV_T * b * 1e3; // eV → meV
            phase[i * n_b + j] = topological_phase_index(e_z, delta, mu_mev);
        }
    }
    phase
}

// ---------------------------------------------------------------------------
// SPECULATIVE — topological magnetoelectric
// ---------------------------------------------------------------------------

/// **SPECULATIVE**: Topological magnetoelectric polarization.
///
/// P = (e^2 / 2 pi h) * theta * B, with theta = pi for TI.
///
/// Returns C/m^2.
pub fn topological_magnetoelectric_polarization(b_tesla: f64, theta: f64) -> f64 {
    (E_CHARGE * E_CHARGE / (2.0 * PI * H_PLANCK)) * theta * b_tesla
}

/// **SPECULATIVE**: Chern number estimate.
///
/// C = sign(E_Z^2 - Delta^2 - mu^2) / 2
pub fn chern_number_estimate(delta_mev: f64, zeeman_mev: f64, mu_mev: f64) -> f64 {
    let disc = zeeman_mev * zeeman_mev - delta_mev * delta_mev - mu_mev * mu_mev;
    if disc.abs() < 1e-10 {
        0.0
    } else {
        0.5 * disc.signum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proximity_bulk_sc_unity() {
        let z = vec![-50.0, -10.0, -1.0];
        let profile = proximity_decay_profile(&z, 100.0, 0.8);
        for &f in &profile {
            assert!((f - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_proximity_interface_transparency() {
        let z = vec![0.0];
        let profile = proximity_decay_profile(&z, 100.0, 0.8);
        assert!((profile[0] - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_proximity_monotonic_decay() {
        let z: Vec<f64> = (0..100).map(|i| i as f64 * 3.0).collect();
        let profile = proximity_decay_profile(&z, 100.0, 0.8);
        for w in profile.windows(2) {
            assert!(w[1] <= w[0] + 1e-15, "not monotonic: {} > {}", w[1], w[0]);
        }
    }

    #[test]
    fn test_gap_3d_shape() {
        let gap_2d = vec![3.0; 16]; // 4x4
        let config = ProximityConfig {
            n_z_layers: 10,
            ..Default::default()
        };
        let result = compute_gap_3d(&gap_2d, 4, &config);
        assert_eq!(result.gap_3d.len(), 10 * 16);
        assert_eq!(result.z_coords.len(), 10);
        assert_eq!(result.decay_profile.len(), 10);
    }

    #[test]
    fn test_dirac_dispersion_linear() {
        let e1 = dirac_dispersion(0.1, DEFAULT_VF);
        let e2 = dirac_dispersion(0.2, DEFAULT_VF);
        // Should be approximately linear: e2 ≈ 2*e1
        assert!((e2 / e1 - 2.0).abs() < 0.01, "e1={}, e2={}", e1, e2);
    }

    #[test]
    fn test_bessel_j0_at_zero() {
        assert!((bessel_j0(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bessel_j0_known_values() {
        // J_0(2.4048) ≈ 0 (first zero)
        assert!(bessel_j0(2.4048).abs() < 0.001);
    }

    #[test]
    fn test_majorana_density_non_negative() {
        let result = majorana_probability_density(
            16, 200.0, &[[0.0, 0.0]], DEFAULT_XI_M, DEFAULT_KF,
        );
        for &v in &result.probability_density {
            assert!(v >= 0.0, "negative density: {}", v);
        }
    }

    #[test]
    fn test_majorana_no_vortices() {
        let result = majorana_probability_density(16, 200.0, &[], DEFAULT_XI_M, DEFAULT_KF);
        assert_eq!(result.n_vortices_with_mzm, 0);
        for &v in &result.probability_density {
            assert!(v.abs() < 1e-15);
        }
    }

    #[test]
    fn test_topological_phase_trivial() {
        // E_Z = 1 meV, Delta = 3 meV → trivial
        assert_eq!(topological_phase_index(1.0, 3.0, 0.0), 0);
    }

    #[test]
    fn test_topological_phase_topological() {
        // E_Z = 5 meV, Delta = 3 meV → topological
        assert_eq!(topological_phase_index(5.0, 3.0, 0.0), 1);
    }

    #[test]
    fn test_chern_number_topological() {
        let c = chern_number_estimate(3.0, 5.0, 0.0);
        assert!((c - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_chern_number_trivial() {
        let c = chern_number_estimate(3.0, 1.0, 0.0);
        assert!((c - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_magnetoelectric_positive() {
        let p = topological_magnetoelectric_polarization(1.0, PI);
        assert!(p > 0.0);
    }
}
