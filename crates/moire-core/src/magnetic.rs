//! Magnetic field effects on moire-modulated Cooper-pair density.
//!
//! Implements Abrikosov vortex lattice generation, Ginzburg-Landau vortex core
//! profiles, Zeeman splitting, flux quantization, Meissner screening currents,
//! and speculative tertiary effects.
//!
//! Established physics unless marked **SPECULATIVE**.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Magnetic flux quantum h/(2e) in Weber.
const PHI_0: f64 = 2.0678e-15;
/// Bohr magneton in meV/T.
const MU_B_MEV_T: f64 = 5.788e-2;
/// Angstrom → metre.
const ANG_TO_M: f64 = 1e-10;
/// Default g-factor for topological surface states.
const DEFAULT_G_FACTOR: f64 = 30.0;
/// Upper critical field for FeTe in Tesla (used in tests).
#[cfg(test)]
const DEFAULT_BC2: f64 = 47.0;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Configuration for applied magnetic field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MagneticFieldConfig {
    /// In-plane field Bx (Tesla).
    pub bx: f64,
    /// In-plane field By (Tesla).
    pub by: f64,
    /// Perpendicular field Bz (Tesla).
    pub bz: f64,
}

impl Default for MagneticFieldConfig {
    fn default() -> Self {
        Self {
            bx: 0.0,
            by: 0.0,
            bz: 0.0,
        }
    }
}

/// Result of Abrikosov vortex lattice computation.
#[derive(Debug, Clone)]
pub struct VortexLatticeResult {
    /// Vortex core positions in Angstrom, (x, y) pairs.
    pub vortex_positions: Vec<[f64; 2]>,
    /// Vortex lattice period in Angstrom.
    pub vortex_period: f64,
    /// Flux per moire unit cell in units of Phi_0.
    pub flux_per_moire_cell: f64,
    /// Whether a_v ≈ L_m within 5%.
    pub is_commensurate: bool,
    /// Commensuration field in Tesla.
    pub commensuration_field: f64,
    /// Order parameter suppression field, row-major NxN, values in [0,1].
    pub suppression_field: Vec<f64>,
    /// Grid resolution.
    pub resolution: usize,
}

/// Result of Zeeman effect computation.
#[derive(Debug, Clone)]
pub struct ZeemanResult {
    /// Zeeman splitting energy in meV.
    pub zeeman_energy: f64,
    /// Pauli paramagnetic limiting field in Tesla.
    pub pauli_limit_field: f64,
    /// Depairing ratio |B_parallel| / B_P.
    pub depairing_ratio: f64,
}

/// Local magnetometric response.
#[derive(Debug, Clone)]
pub struct MagnetometricResponse {
    /// Local susceptibility map, row-major.
    pub susceptibility_map: Vec<f64>,
    /// Screening current x-component, row-major.
    pub screening_jx: Vec<f64>,
    /// Screening current y-component, row-major.
    pub screening_jy: Vec<f64>,
    /// Grid resolution.
    pub resolution: usize,
}

// ---------------------------------------------------------------------------
// Established physics — Abrikosov vortex lattice
// ---------------------------------------------------------------------------

/// Triangular Abrikosov vortex lattice period.
///
/// a_v = sqrt(2 * Phi_0 / (sqrt(3) * |Bz|))
///
/// Returns period in Angstrom.  Returns f64::INFINITY for Bz ≈ 0.
pub fn vortex_lattice_period(bz: f64) -> f64 {
    let bz_abs = bz.abs();
    if bz_abs < 1e-15 {
        return f64::INFINITY;
    }
    let a_v_m = (2.0 * PHI_0 / (3.0_f64.sqrt() * bz_abs)).sqrt();
    a_v_m / ANG_TO_M
}

/// Field at which vortex lattice period equals moire period.
///
/// B_comm = 2 * Phi_0 / (sqrt(3) * L_m^2)
pub fn commensuration_field(moire_period_ang: f64) -> f64 {
    let l_m = moire_period_ang * ANG_TO_M;
    if l_m < 1e-30 || !moire_period_ang.is_finite() {
        return f64::INFINITY;
    }
    2.0 * PHI_0 / (3.0_f64.sqrt() * l_m * l_m)
}

/// Generate triangular vortex lattice positions within viewport.
///
/// Returns Vec of [x, y] positions in Angstrom.
pub fn generate_vortex_positions(bz: f64, physical_extent: f64) -> Vec<[f64; 2]> {
    let a_v = vortex_lattice_period(bz);
    if !a_v.is_finite() || a_v <= 0.0 {
        return Vec::new();
    }

    let half = physical_extent / 2.0;
    let a2y = a_v * 3.0_f64.sqrt() / 2.0;

    let n_cols = (physical_extent / a_v).ceil() as i32 + 2;
    let n_rows = (physical_extent / a2y).ceil() as i32 + 2;

    let mut positions = Vec::new();
    for row in -n_rows..=n_rows {
        let y = row as f64 * a2y;
        let x_offset = if row % 2 != 0 { a_v / 2.0 } else { 0.0 };
        for col in -n_cols..=n_cols {
            let x = col as f64 * a_v + x_offset;
            if x >= -half && x <= half && y >= -half && y <= half {
                positions.push([x, y]);
            }
        }
    }
    positions
}

/// Order parameter suppression from vortex cores.
///
/// suppression(r) = Product_v tanh(|r - r_v| / xi)
///
/// Returns row-major NxN array of values in [0, 1].
pub fn vortex_suppression_field(
    resolution: usize,
    physical_extent: f64,
    vortex_positions: &[[f64; 2]],
    coherence_length: f64,
) -> Vec<f64> {
    let n = resolution;
    let mut suppression = vec![1.0_f64; n * n];

    if vortex_positions.is_empty() {
        return suppression;
    }

    let cutoff = 5.0 * coherence_length;

    for iy in 0..n {
        let y = (iy as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
        for ix in 0..n {
            let x = (ix as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
            let idx = iy * n + ix;

            for &[vx, vy] in vortex_positions {
                let dx = x - vx;
                let dy = y - vy;
                let dist = (dx * dx + dy * dy).sqrt();
                if dist < cutoff {
                    suppression[idx] *= (dist / coherence_length).tanh();
                }
            }
        }
    }
    suppression
}

/// Combine moire gap modulation with vortex suppression.
///
/// Delta_total = Delta_moire * suppression  (element-wise).
pub fn combined_gap_with_vortices(gap: &[f64], suppression: &[f64]) -> Vec<f64> {
    gap.iter()
        .zip(suppression.iter())
        .map(|(&g, &s)| g * s)
        .collect()
}

// ---------------------------------------------------------------------------
// Established physics — flux quantization and Zeeman
// ---------------------------------------------------------------------------

/// Flux per moire unit cell: n = |Bz| * A_moire / Phi_0.
pub fn flux_per_moire_cell(bz: f64, moire_period_ang: f64) -> f64 {
    let l_m = moire_period_ang * ANG_TO_M;
    let a_moire = 3.0_f64.sqrt() / 2.0 * l_m * l_m;
    bz.abs() * a_moire / PHI_0
}

/// Zeeman energy from in-plane field: E_Z = g * mu_B * |B_parallel|.
///
/// Returns meV.
pub fn zeeman_energy(bx: f64, by: f64, g_factor: f64) -> f64 {
    let b_par = (bx * bx + by * by).sqrt();
    g_factor * MU_B_MEV_T * b_par
}

/// Pauli paramagnetic limiting field: B_P = Delta / (sqrt(2) * mu_B).
///
/// Returns Tesla.
pub fn pauli_limiting_field(delta_avg_mev: f64) -> f64 {
    if delta_avg_mev <= 0.0 {
        return f64::INFINITY;
    }
    delta_avg_mev / (2.0_f64.sqrt() * MU_B_MEV_T)
}

/// Compute full Zeeman result.
pub fn compute_zeeman(config: &MagneticFieldConfig, delta_avg_mev: f64) -> ZeemanResult {
    let e_z = zeeman_energy(config.bx, config.by, DEFAULT_G_FACTOR);
    let b_p = pauli_limiting_field(delta_avg_mev);
    let b_par = (config.bx * config.bx + config.by * config.by).sqrt();
    let ratio = if b_p.is_finite() && b_p > 0.0 {
        b_par / b_p
    } else {
        0.0
    };
    ZeemanResult {
        zeeman_energy: e_z,
        pauli_limit_field: b_p,
        depairing_ratio: ratio,
    }
}

// ---------------------------------------------------------------------------
// Established physics — screening currents
// ---------------------------------------------------------------------------

/// Meissner screening supercurrent density around vortex cores.
///
/// Returns (jx, jy) as row-major Vec<f64>, normalised to peak = 1.
pub fn screening_currents(
    resolution: usize,
    physical_extent: f64,
    vortex_positions: &[[f64; 2]],
    lambda_l: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = resolution;
    let mut jx = vec![0.0_f64; n * n];
    let mut jy = vec![0.0_f64; n * n];

    if vortex_positions.is_empty() {
        return (jx, jy);
    }

    for iy in 0..n {
        let y = (iy as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
        for ix in 0..n {
            let x = (ix as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
            let idx = iy * n + ix;

            for &[vx, vy] in vortex_positions {
                let dx = x - vx;
                let dy = y - vy;
                let r = (dx * dx + dy * dy).sqrt();
                let r_safe = if r < 1e-10 { 1.0 } else { r };
                let mag = (-r / lambda_l).exp() / r_safe;
                jx[idx] += -dy / r_safe * mag;
                jy[idx] += dx / r_safe * mag;
            }
        }
    }

    // Normalise to peak = 1
    let mut peak = 0.0_f64;
    for i in 0..n * n {
        let m = (jx[i] * jx[i] + jy[i] * jy[i]).sqrt();
        if m > peak {
            peak = m;
        }
    }
    if peak > 1e-30 {
        for i in 0..n * n {
            jx[i] /= peak;
            jy[i] /= peak;
        }
    }

    (jx, jy)
}

// ---------------------------------------------------------------------------
// SPECULATIVE — local susceptibility
// ---------------------------------------------------------------------------

/// **SPECULATIVE**: Local magnetic susceptibility chi(r).
///
/// chi(r) ~ -(1 - (Delta(r)/Delta_max)^2), normalised to [-1, 0].
pub fn local_susceptibility(gap_field: &[f64]) -> Vec<f64> {
    let gap_max = gap_field.iter().cloned().fold(0.0_f64, |a, b| a.max(b.abs()));
    if gap_max < 1e-30 {
        return vec![0.0; gap_field.len()];
    }
    gap_field
        .iter()
        .map(|&g| {
            let norm = g / gap_max;
            -(1.0 - norm * norm)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SPECULATIVE — tertiary physics
// ---------------------------------------------------------------------------

/// **SPECULATIVE**: Beating pattern from moire and vortex lattice interference.
///
/// L_beat = L_m * L_v / |L_m - L_v|
///
/// Returns row-major NxN pattern normalised to [0, 1].
pub fn moire_vortex_beating(
    moire_period: f64,
    vortex_period: f64,
    resolution: usize,
    physical_extent: f64,
) -> Vec<f64> {
    let n = resolution;
    let diff = (moire_period - vortex_period).abs();
    if diff < 1e-10 {
        return vec![1.0; n * n];
    }

    let l_beat = moire_period * vortex_period / diff;
    let q_beat = 2.0 * PI / l_beat;

    let mut pattern = vec![0.0_f64; n * n];
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;

    for iy in 0..n {
        let y = (iy as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
        for ix in 0..n {
            let x = (ix as f64 / (n - 1).max(1) as f64 - 0.5) * physical_extent;
            let mut val = 0.0;
            for k in 0..6 {
                let angle = k as f64 * PI / 3.0;
                val += (q_beat * angle.cos() * x + q_beat * angle.sin() * y).cos();
            }
            pattern[iy * n + ix] = val;
            if val < min_val { min_val = val; }
            if val > max_val { max_val = val; }
        }
    }

    let range = max_val - min_val;
    if range > 1e-15 {
        for v in pattern.iter_mut() {
            *v = (*v - min_val) / range;
        }
    } else {
        for v in pattern.iter_mut() {
            *v = 0.5;
        }
    }
    pattern
}

/// **SPECULATIVE**: CPDM amplitude modulated by applied field.
///
/// A(B) = A0 * (1 - |Bz|/Bc2)
pub fn field_tunable_cpdm(
    cpdm_amplitude_zero: f64,
    bz: f64,
    bc2: f64,
) -> f64 {
    let ratio = if bc2 > 0.0 {
        (bz.abs() / bc2).min(1.0)
    } else {
        0.0
    };
    cpdm_amplitude_zero * (1.0 - ratio)
}

/// **SPECULATIVE**: Cosine pinning energy for vortex-moire commensuration.
///
/// E_pin ~ cos(2 pi L_m / L_v)
pub fn commensuration_pinning_energy(moire_period: f64, vortex_period: f64) -> f64 {
    if vortex_period < 1e-10 || !vortex_period.is_finite() {
        return 0.0;
    }
    (2.0 * PI * moire_period / vortex_period).cos()
}

// ---------------------------------------------------------------------------
// Full computation pipeline
// ---------------------------------------------------------------------------

/// Compute complete magnetic field effects for a moire system.
///
/// Takes the moire result and magnetic config, returns vortex lattice and
/// all derived quantities.
pub fn compute_magnetic_effects(
    config: &MagneticFieldConfig,
    moire_period: f64,
    resolution: usize,
    physical_extent: f64,
    coherence_length: f64,
) -> VortexLatticeResult {
    let vortex_period = vortex_lattice_period(config.bz);
    let positions = generate_vortex_positions(config.bz, physical_extent);
    let suppression =
        vortex_suppression_field(resolution, physical_extent, &positions, coherence_length);

    let flux = if moire_period.is_finite() {
        flux_per_moire_cell(config.bz, moire_period)
    } else {
        0.0
    };

    let comm_field = if moire_period.is_finite() {
        commensuration_field(moire_period)
    } else {
        f64::INFINITY
    };

    let is_comm = if vortex_period.is_finite() && moire_period.is_finite() && moire_period > 0.0 {
        let ratio = vortex_period / moire_period;
        (ratio - 1.0).abs() < 0.05
    } else {
        false
    };

    VortexLatticeResult {
        vortex_positions: positions,
        vortex_period,
        flux_per_moire_cell: flux,
        is_commensurate: is_comm,
        commensuration_field: comm_field,
        suppression_field: suppression,
        resolution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vortex_period_at_1_tesla() {
        let p = vortex_lattice_period(1.0);
        // Expected ~489 Angstrom
        assert!(p > 480.0 && p < 500.0, "period = {}", p);
    }

    #[test]
    fn test_vortex_period_scales_inverse_sqrt_b() {
        let p1 = vortex_lattice_period(1.0);
        let p4 = vortex_lattice_period(4.0);
        // p4 should be p1/2 (sqrt(4) = 2)
        assert!((p4 - p1 / 2.0).abs() < 1.0, "p1={}, p4={}", p1, p4);
    }

    #[test]
    fn test_zero_field_infinite_period() {
        assert!(vortex_lattice_period(0.0).is_infinite());
    }

    #[test]
    fn test_zero_field_no_vortices() {
        let pos = generate_vortex_positions(0.0, 100.0);
        assert!(pos.is_empty());
    }

    #[test]
    fn test_vortex_positions_nonempty_at_1t() {
        let pos = generate_vortex_positions(1.0, 1000.0);
        assert!(!pos.is_empty());
    }

    #[test]
    fn test_suppression_at_core_near_zero() {
        // Place single vortex at origin, check center of grid
        let n = 101;
        let extent = 200.0;
        let xi = 20.0;
        let positions = vec![[0.0, 0.0]];
        let supp = vortex_suppression_field(n, extent, &positions, xi);
        // Center pixel
        let center = supp[50 * n + 50];
        assert!(center < 0.05, "center suppression = {}", center);
    }

    #[test]
    fn test_suppression_far_from_core() {
        let n = 101;
        let extent = 2000.0;
        let xi = 20.0;
        let positions = vec![[0.0, 0.0]];
        let supp = vortex_suppression_field(n, extent, &positions, xi);
        // Corner pixel
        let corner = supp[0];
        assert!(corner > 0.99, "corner suppression = {}", corner);
    }

    #[test]
    fn test_flux_quantization() {
        let l_m = 100.0; // Angstrom
        // B giving exactly 1 flux quantum
        let a_m = 3.0_f64.sqrt() / 2.0 * (l_m * ANG_TO_M).powi(2);
        let b = PHI_0 / a_m;
        let n = flux_per_moire_cell(b, l_m);
        assert!((n - 1.0).abs() < 0.01, "flux/cell = {}", n);
    }

    #[test]
    fn test_zeeman_zero_field() {
        assert_eq!(zeeman_energy(0.0, 0.0, DEFAULT_G_FACTOR), 0.0);
    }

    #[test]
    fn test_pauli_limit_reasonable() {
        // For Delta = 3 meV, B_P ~ 37 T
        let bp = pauli_limiting_field(3.0);
        assert!(bp > 30.0 && bp < 50.0, "B_P = {}", bp);
    }

    #[test]
    fn test_combined_gap_shape() {
        let gap = vec![3.0; 100];
        let supp = vec![0.5; 100];
        let combined = combined_gap_with_vortices(&gap, &supp);
        assert_eq!(combined.len(), 100);
        for &v in &combined {
            assert!((v - 1.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_commensuration_field_large_for_small_period() {
        // L_m = 36.7 A (Sb2Te3/FeTe)
        let b = commensuration_field(36.7);
        assert!(b > 100.0, "B_comm = {}", b);
    }

    #[test]
    fn test_field_tunable_cpdm_zero_field() {
        let a = field_tunable_cpdm(0.8, 0.0, DEFAULT_BC2);
        assert!((a - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_field_tunable_cpdm_at_bc2() {
        let a = field_tunable_cpdm(0.8, DEFAULT_BC2, DEFAULT_BC2);
        assert!(a.abs() < 1e-10);
    }

    #[test]
    fn test_pinning_energy_commensurate() {
        // L_m = L_v → ratio = 1 → cos(2pi) = 1
        let e = commensuration_pinning_energy(100.0, 100.0);
        assert!((e - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_local_susceptibility_uniform_gap() {
        let gap = vec![3.0; 16];
        let chi = local_susceptibility(&gap);
        // All values should be 0 (since (3/3)^2 = 1, chi = -(1-1) = 0)
        for &c in &chi {
            assert!(c.abs() < 1e-10, "chi = {}", c);
        }
    }
}
