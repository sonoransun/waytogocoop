//! Curved graphene sheets: Monge-gauge strain and pseudo-magnetic fields
//! (**SPECULATIVE** gap coupling).
//!
//! Established physics unless marked **SPECULATIVE**.
//!
//! Literature basis for strain-induced pseudo-gauge fields in graphene:
//!   - Guinea, Katsnelson & Geim, Nat. Phys. 6, 30 (2010) — strain engineering
//!   - Levy et al., Science 329, 544 (2010) — 300 T pseudo-fields in nanobubbles
//!   - Vozmediano, Katsnelson & Guinea, Phys. Rep. 496, 109 (2010) — gauge
//!     fields in graphene, Monge-gauge strain eps_ij = (dh/dx_i)(dh/dx_j)/2
//!
//! Caveat: the pseudo-field is valley-antisymmetric (B_K = -B_K'), so
//! time-reversal symmetry is preserved; it pair-breaks intervalley singlets
//! only through the **SPECULATIVE** local suppression model used here.
//!
//! Mirrors `src/waytogocoop/computation/curvature.py` in the Python stack —
//! changes here must land in both.

use rayon::prelude::*;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Reduced Planck constant in J*s.
const HBAR_J_S: f64 = 1.054571817e-34;
/// Elementary charge in Coulomb.
const E_CHARGE_C: f64 = 1.602176634e-19;
/// Angstrom → metre.
const ANG_TO_M: f64 = 1e-10;
/// Graphene carbon-carbon bond length in Angstrom.
const A_CC_GRAPHENE: f64 = 1.42;
/// Dimensionless electron-phonon coupling beta = -d(ln t)/d(ln a).
pub const GRAPHENE_BETA_DEFAULT: f64 = 3.0;
/// Pseudo-field scale at which the gap is fully suppressed, in Tesla.
pub const B_PAIRBREAK_DEFAULT: f64 = 10.0;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Out-of-plane deformation geometry of the graphene sheet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CurvatureGeometry {
    #[default]
    Flat,
    GaussianBump,
    SinusoidalRipple,
    CylindricalBend,
    SphericalCap,
}

impl CurvatureGeometry {
    pub const ALL: [CurvatureGeometry; 5] = [
        CurvatureGeometry::Flat,
        CurvatureGeometry::GaussianBump,
        CurvatureGeometry::SinusoidalRipple,
        CurvatureGeometry::CylindricalBend,
        CurvatureGeometry::SphericalCap,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            CurvatureGeometry::Flat => "Flat (planar)",
            CurvatureGeometry::GaussianBump => "Gaussian bump",
            CurvatureGeometry::SinusoidalRipple => "Sinusoidal ripple",
            CurvatureGeometry::CylindricalBend => "Cylindrical bend",
            CurvatureGeometry::SphericalCap => "Spherical cap",
        }
    }
}

/// Configuration for a curvature / pseudo-magnetic-field computation.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurvatureConfig {
    pub geometry: CurvatureGeometry,
    /// Height amplitude h0 in Angstrom (bump and ripple).
    pub amplitude: f64,
    /// Gaussian bump width in Angstrom.
    pub sigma: f64,
    /// Ripple wavelength in Angstrom.
    pub wavelength: f64,
    /// Bend / cap radius of curvature in Angstrom.
    pub radius: f64,
    /// In-plane orientation of ripple / bend axis in degrees.
    pub orientation_deg: f64,
    /// Electron-phonon coupling beta.
    pub beta: f64,
    /// Pair-breaking field scale in Tesla (SPECULATIVE gap coupling).
    pub b_pairbreak: f64,
    /// Grid resolution (NxN pixels).
    pub resolution: usize,
    /// Viewport size in Angstroms.
    pub physical_extent: f64,
    /// Valley index: +1 for K, -1 for K' (the pseudo-field is valley-odd).
    #[serde(default = "default_valley")]
    pub valley: i32,
}

fn default_valley() -> i32 {
    1
}

impl Default for CurvatureConfig {
    fn default() -> Self {
        Self {
            geometry: CurvatureGeometry::Flat,
            amplitude: 5.0,
            sigma: 50.0,
            wavelength: 100.0,
            radius: 1000.0,
            orientation_deg: 0.0,
            beta: GRAPHENE_BETA_DEFAULT,
            b_pairbreak: B_PAIRBREAK_DEFAULT,
            resolution: 256,
            physical_extent: 400.0,
            valley: default_valley(),
        }
    }
}

/// Result of a curvature computation.  All grids are row-major NxN.
#[derive(Debug, Clone)]
pub struct CurvatureResult {
    /// Height field h(r) in Angstrom.
    pub height: Vec<f64>,
    /// Monge-gauge strain component eps_xx.
    pub strain_xx: Vec<f64>,
    /// Monge-gauge strain component eps_yy.
    pub strain_yy: Vec<f64>,
    /// Monge-gauge strain component eps_xy.
    pub strain_xy: Vec<f64>,
    /// Pseudo-magnetic field in Tesla, signed, for the configured valley
    /// (the opposite valley sees the opposite sign).
    pub pseudo_field: Vec<f64>,
    /// **SPECULATIVE**: local gap suppression factor in [0, 1].
    pub gap_suppression: Vec<f64>,
    pub resolution: usize,
    pub physical_extent: f64,
    /// Maximum |B| over the (border-clamped) pseudo-field, in Tesla.
    pub max_abs_field: f64,
}

// ---------------------------------------------------------------------------
// Numerics
// ---------------------------------------------------------------------------

/// Central-difference gradient of a row-major NxN field, matching
/// `np.gradient(field, dx, edge_order=1)`: interior points use
/// (f[i+1] - f[i-1]) / (2*dx), edges use one-sided differences.
/// Returns (d/dx along columns, d/dy along rows).
fn gradient_2d(field: &[f64], n: usize, dx: f64) -> (Vec<f64>, Vec<f64>) {
    let d_dx: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|iy| {
            (0..n).map(move |ix| {
                let idx = iy * n + ix;
                if ix == 0 {
                    (field[idx + 1] - field[idx]) / dx
                } else if ix == n - 1 {
                    (field[idx] - field[idx - 1]) / dx
                } else {
                    (field[idx + 1] - field[idx - 1]) / (2.0 * dx)
                }
            })
        })
        .collect();

    let d_dy: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|iy| {
            (0..n).map(move |ix| {
                let idx = iy * n + ix;
                if iy == 0 {
                    (field[idx + n] - field[idx]) / dx
                } else if iy == n - 1 {
                    (field[idx] - field[idx - n]) / dx
                } else {
                    (field[idx + n] - field[idx - n]) / (2.0 * dx)
                }
            })
        })
        .collect();

    (d_dx, d_dy)
}

/// **SPECULATIVE**: In-plane displacement induced by the out-of-plane
/// corrugation, u = -h * grad(h) / 2 — the leading-order in-plane relaxation
/// of a corrugated membrane, suitable as a warp input for
/// `graphene::compute_graphene_stack_v2`.  `height` is a row-major NxN field
/// in Angstrom, `dx` the grid spacing in Angstrom; returns (u_x, u_y) in
/// Angstrom.
pub fn displacement_field(height: &[f64], n: usize, dx: f64) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(height.len(), n * n, "height must be a row-major n*n field");
    let (dh_dx, dh_dy) = gradient_2d(height, n, dx);
    let u_x = height
        .iter()
        .zip(dh_dx.iter())
        .map(|(h, g)| -0.5 * h * g)
        .collect();
    let u_y = height
        .iter()
        .zip(dh_dy.iter())
        .map(|(h, g)| -0.5 * h * g)
        .collect();
    (u_x, u_y)
}

/// Overwrite the 2-pixel border with the nearest interior row/column: the
/// stacked one-sided finite differences are unreliable there.  Rows first,
/// then columns, matching the Python implementation.
fn clamp_border(field: &mut [f64], n: usize) {
    for iy in [0, 1, n - 2, n - 1] {
        let src = if iy < 2 { 2 } else { n - 3 };
        for ix in 0..n {
            field[iy * n + ix] = field[src * n + ix];
        }
    }
    for ix in [0, 1, n - 2, n - 1] {
        let src = if ix < 2 { 2 } else { n - 3 };
        for row in field.chunks_exact_mut(n) {
            row[ix] = row[src];
        }
    }
}

// ---------------------------------------------------------------------------
// SPECULATIVE — gap coupling
// ---------------------------------------------------------------------------

/// **SPECULATIVE**: Local gap suppression from the pseudo-magnetic field:
/// suppression = clamp(1 - |B| / B_pairbreak, 0, 1) per pixel.
pub fn gap_suppression_field(pseudo_field: &[f64], b_pairbreak: f64) -> Vec<f64> {
    pseudo_field
        .iter()
        .map(|b| (1.0 - b.abs() / b_pairbreak).clamp(0.0, 1.0))
        .collect()
}

// ---------------------------------------------------------------------------
// Full computation pipeline
// ---------------------------------------------------------------------------

/// Compute height, Monge-gauge strain, pseudo-magnetic field and speculative
/// gap suppression for a curved graphene sheet.
///
/// Strain eps_ij = (dh/dx_i)(dh/dx_j) / 2; pseudo-gauge field
/// A = (hbar*beta / (2*e*a_cc)) * (eps_xx - eps_yy, -2*eps_xy);
/// B = dA_y/dx - dA_x/dy (K valley).
pub fn compute_curvature(cfg: &CurvatureConfig) -> Result<CurvatureResult, String> {
    if cfg.resolution < 8 {
        return Err(format!("resolution must be >= 8, got {}", cfg.resolution));
    }
    if cfg.physical_extent <= 0.0 {
        return Err(format!(
            "physical_extent must be positive, got {}",
            cfg.physical_extent
        ));
    }
    if cfg.amplitude < 0.0 {
        return Err(format!(
            "amplitude must be non-negative, got {}",
            cfg.amplitude
        ));
    }
    if cfg.sigma <= 0.0 {
        return Err(format!("sigma must be positive, got {}", cfg.sigma));
    }
    if cfg.wavelength <= 0.0 {
        return Err(format!(
            "wavelength must be positive, got {}",
            cfg.wavelength
        ));
    }
    if cfg.radius <= 0.0 {
        return Err(format!("radius must be positive, got {}", cfg.radius));
    }
    if cfg.b_pairbreak <= 0.0 {
        return Err(format!(
            "b_pairbreak must be positive, got {}",
            cfg.b_pairbreak
        ));
    }
    if cfg.valley != 1 && cfg.valley != -1 {
        return Err(format!("valley must be +1 or -1, got {}", cfg.valley));
    }

    let n = cfg.resolution;
    let extent = cfg.physical_extent;
    let dx = extent / (n - 1) as f64;

    let geometry = cfg.geometry;
    let h0 = cfg.amplitude;
    let sigma = cfg.sigma;
    let wavelength = cfg.wavelength;
    let radius = cfg.radius;
    let phi = cfg.orientation_deg.to_radians();
    let (cos_phi, sin_phi) = (phi.cos(), phi.sin());

    let height: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|iy| {
            let y = (iy as f64 / (n - 1) as f64 - 0.5) * extent;
            (0..n).map(move |ix| {
                let x = (ix as f64 / (n - 1) as f64 - 0.5) * extent;
                match geometry {
                    CurvatureGeometry::Flat => 0.0,
                    CurvatureGeometry::GaussianBump => {
                        h0 * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
                    }
                    CurvatureGeometry::SinusoidalRipple => {
                        let u = x * cos_phi + y * sin_phi;
                        h0 * (2.0 * PI * u / wavelength).sin()
                    }
                    CurvatureGeometry::CylindricalBend => {
                        let u = x * cos_phi + y * sin_phi;
                        let v = u.abs().min(radius);
                        radius - (radius * radius - v * v).sqrt()
                    }
                    CurvatureGeometry::SphericalCap => {
                        (radius * radius - (x * x + y * y)).max(0.0).sqrt()
                    }
                }
            })
        })
        .collect();

    let (dh_dx, dh_dy) = gradient_2d(&height, n, dx);
    let strain_xx: Vec<f64> = dh_dx.iter().map(|g| 0.5 * g * g).collect();
    let strain_yy: Vec<f64> = dh_dy.iter().map(|g| 0.5 * g * g).collect();
    let strain_xy: Vec<f64> = dh_dx
        .iter()
        .zip(dh_dy.iter())
        .map(|(gx, gy)| 0.5 * gx * gy)
        .collect();

    // Gauge field in T*m: A_x = pref*(eps_xx - eps_yy), A_y = pref*(-2*eps_xy)
    let pref = HBAR_J_S * cfg.beta / (2.0 * E_CHARGE_C * A_CC_GRAPHENE * ANG_TO_M);
    let a_x: Vec<f64> = strain_xx
        .iter()
        .zip(strain_yy.iter())
        .map(|(xx, yy)| pref * (xx - yy))
        .collect();
    let a_y: Vec<f64> = strain_xy.iter().map(|xy| pref * (-2.0 * xy)).collect();

    let (day_dx, _) = gradient_2d(&a_y, n, dx);
    let (_, dax_dy) = gradient_2d(&a_x, n, dx);
    let valley = cfg.valley as f64;
    let mut pseudo_field: Vec<f64> = day_dx
        .iter()
        .zip(dax_dy.iter())
        .map(|(dy_val, dx_val)| valley * (dy_val - dx_val) / ANG_TO_M)
        .collect();
    clamp_border(&mut pseudo_field, n);

    let gap_suppression = gap_suppression_field(&pseudo_field, cfg.b_pairbreak);
    let max_abs_field = pseudo_field.iter().fold(0.0_f64, |acc, b| acc.max(b.abs()));

    Ok(CurvatureResult {
        height,
        strain_xx,
        strain_yy,
        strain_xy,
        pseudo_field,
        gap_suppression,
        resolution: n,
        physical_extent: extent,
        max_abs_field,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config(geometry: CurvatureGeometry) -> CurvatureConfig {
        CurvatureConfig {
            geometry,
            resolution: 200,
            physical_extent: 200.0,
            ..Default::default()
        }
    }

    fn ripple_config(orientation_deg: f64) -> CurvatureConfig {
        CurvatureConfig {
            amplitude: 2.0,
            wavelength: 100.0,
            orientation_deg,
            ..test_config(CurvatureGeometry::SinusoidalRipple)
        }
    }

    fn bump_config() -> CurvatureConfig {
        CurvatureConfig {
            amplitude: 5.0,
            sigma: 50.0,
            ..test_config(CurvatureGeometry::GaussianBump)
        }
    }

    #[test]
    fn test_flat_all_zero() {
        let result = compute_curvature(&test_config(CurvatureGeometry::Flat)).unwrap();
        assert!(result.height.iter().all(|&v| v == 0.0));
        assert!(result.strain_xx.iter().all(|&v| v == 0.0));
        assert!(result.strain_yy.iter().all(|&v| v == 0.0));
        assert!(result.strain_xy.iter().all(|&v| v == 0.0));
        assert!(result.pseudo_field.iter().all(|&v| v == 0.0));
        assert!(result.gap_suppression.iter().all(|&v| v == 1.0));
        assert_eq!(result.max_abs_field, 0.0);
    }

    #[test]
    fn test_ripple_zigzag_no_field() {
        // A ripple along the zigzag direction (phi = 0) has eps_yy = eps_xy = 0
        // and A_x depending on x only, so B vanishes identically.
        let result = compute_curvature(&ripple_config(0.0)).unwrap();
        assert!(result.max_abs_field < 1e-8, "max = {}", result.max_abs_field);
    }

    #[test]
    fn test_ripple_sin_3phi_dependence() {
        let max_30 = compute_curvature(&ripple_config(30.0)).unwrap().max_abs_field;
        for phi in [10.0_f64, 20.0] {
            let max_phi = compute_curvature(&ripple_config(phi)).unwrap().max_abs_field;
            let expected = (3.0 * phi.to_radians()).sin();
            let ratio = max_phi / max_30;
            assert!(
                (ratio - expected).abs() / expected < 0.05,
                "phi = {}: ratio = {}, expected = {}",
                phi,
                ratio,
                expected
            );
        }
    }

    #[test]
    fn test_ripple_armchair_field_magnitude() {
        let result = compute_curvature(&ripple_config(30.0)).unwrap();
        assert!(
            (result.max_abs_field - 34.49).abs() / 34.49 < 0.05,
            "max = {}",
            result.max_abs_field
        );
    }

    #[test]
    fn test_cylinder_zigzag_no_field() {
        let cfg = CurvatureConfig {
            radius: 1000.0,
            orientation_deg: 0.0,
            ..test_config(CurvatureGeometry::CylindricalBend)
        };
        let result = compute_curvature(&cfg).unwrap();
        assert!(result.max_abs_field < 1e-8, "max = {}", result.max_abs_field);
    }

    #[test]
    fn test_cylinder_rotated_field_magnitude() {
        let cfg = CurvatureConfig {
            radius: 1000.0,
            orientation_deg: 30.0,
            ..test_config(CurvatureGeometry::CylindricalBend)
        };
        let result = compute_curvature(&cfg).unwrap();
        assert!(result.max_abs_field > 1.0, "max = {}", result.max_abs_field);
    }

    #[test]
    fn test_bump_field_magnitude() {
        let result = compute_curvature(&bump_config()).unwrap();
        assert!(
            (result.max_abs_field - 5.70).abs() / 5.70 < 0.05,
            "max = {}",
            result.max_abs_field
        );
    }

    #[test]
    fn test_gentle_cap_much_weaker_than_bump() {
        let cap = CurvatureConfig {
            radius: 2000.0,
            ..test_config(CurvatureGeometry::SphericalCap)
        };
        let cap_max = compute_curvature(&cap).unwrap().max_abs_field;
        let bump_max = compute_curvature(&bump_config()).unwrap().max_abs_field;
        assert!(cap_max > 0.0);
        assert!(
            cap_max < 0.05 * bump_max,
            "cap = {}, bump = {}",
            cap_max,
            bump_max
        );
    }

    #[test]
    fn test_bump_field_mean_near_zero() {
        // The three-fold-symmetric bump field has alternating sign lobes,
        // so the signed mean nearly cancels.
        let result = compute_curvature(&bump_config()).unwrap();
        let mean: f64 =
            result.pseudo_field.iter().sum::<f64>() / result.pseudo_field.len() as f64;
        assert!(
            mean.abs() < 0.02 * result.max_abs_field,
            "mean = {}, max = {}",
            mean,
            result.max_abs_field
        );
    }

    #[test]
    fn test_gap_suppression_bounds() {
        let result = compute_curvature(&ripple_config(30.0)).unwrap();
        assert!(result
            .gap_suppression
            .iter()
            .all(|&v| (0.0..=1.0).contains(&v)));
        let flat = compute_curvature(&test_config(CurvatureGeometry::Flat)).unwrap();
        assert!(flat.gap_suppression.iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_gap_suppression_field_values() {
        let supp = gap_suppression_field(&[0.0, 5.0, 20.0, -20.0], 10.0);
        let expected = [1.0, 0.5, 0.0, 0.0];
        for (s, e) in supp.iter().zip(expected.iter()) {
            assert!((s - e).abs() < 1e-12, "got {}, expected {}", s, e);
        }
    }

    #[test]
    fn test_invalid_inputs() {
        let base = test_config(CurvatureGeometry::Flat);
        let cases = [
            CurvatureConfig { resolution: 4, ..base },
            CurvatureConfig { physical_extent: 0.0, ..base },
            CurvatureConfig { amplitude: -1.0, ..base },
            CurvatureConfig { sigma: 0.0, ..base },
            CurvatureConfig { wavelength: 0.0, ..base },
            CurvatureConfig { radius: 0.0, ..base },
            CurvatureConfig { b_pairbreak: 0.0, ..base },
            CurvatureConfig { valley: 0, ..base },
            CurvatureConfig { valley: 2, ..base },
        ];
        for cfg in &cases {
            assert!(compute_curvature(cfg).is_err(), "cfg = {:?}", cfg);
        }
    }

    #[test]
    fn test_valley_flips_pseudo_field_sign_exactly() {
        let k = compute_curvature(&bump_config()).unwrap();
        let k_prime = compute_curvature(&CurvatureConfig {
            valley: -1,
            ..bump_config()
        })
        .unwrap();
        for (b_k, b_kp) in k.pseudo_field.iter().zip(k_prime.pseudo_field.iter()) {
            assert_eq!(*b_kp, -b_k);
        }
        assert_eq!(k.max_abs_field, k_prime.max_abs_field);
        assert_eq!(k.gap_suppression, k_prime.gap_suppression);
    }

    #[test]
    fn test_displacement_field_flat_zero() {
        let n = 32;
        let height = vec![0.0; n * n];
        let (u_x, u_y) = displacement_field(&height, n, 1.0);
        assert!(u_x.iter().all(|&v| v == 0.0));
        assert!(u_y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_displacement_field_bump_nonzero_and_inward() {
        let result = compute_curvature(&bump_config()).unwrap();
        let n = result.resolution;
        let dx = result.physical_extent / (n - 1) as f64;
        let (u_x, u_y) = displacement_field(&result.height, n, dx);
        let max_u = u_x
            .iter()
            .chain(u_y.iter())
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_u > 1e-3, "max |u| = {}", max_u);
        // u = -h*grad(h)/2 points down-slope: on the +x flank h > 0 and
        // dh/dx < 0, so u_x > 0.
        let iy = n / 2;
        let ix = 3 * n / 4;
        assert!(u_x[iy * n + ix] > 0.0);
    }
}
