use crate::materials::LatticeType;
use nalgebra::Vector2;
use std::f64::consts::PI;

/// Configuration for a moire pattern computation.
#[derive(Debug, Clone)]
pub struct MoireConfig {
    pub substrate_a: f64,
    pub substrate_lattice_type: LatticeType,
    pub overlayer_a: f64,
    pub overlayer_lattice_type: LatticeType,
    /// Twist angle of the overlayer in degrees.
    pub twist_angle_deg: f64,
    /// Grid resolution (NxN pixels).
    pub resolution: usize,
    /// Viewport size in Angstroms.
    pub physical_extent: f64,
}

/// Result of a moire pattern computation.
#[derive(Debug, Clone)]
pub struct MoireResult {
    /// Row-major intensity values, resolution x resolution, normalized to [0, 1].
    pub pattern: Vec<f64>,
    pub resolution: usize,
    pub physical_extent: f64,
    /// Estimated moire period in Angstroms.
    pub moire_period: f64,
    /// Lattice mismatch percentage.
    pub mismatch_percent: f64,
}

/// Compute the 1D moire periodicity from two lattice constants.
/// Lambda = a1*a2 / |a1 - a2|
pub fn moire_periodicity_1d(a1: f64, a2: f64) -> f64 {
    let diff = (a1 - a2).abs();
    if diff < 1e-15 {
        return f64::INFINITY;
    }
    a1 * a2 / diff
}

/// Compute the moire periodicity from a twist angle (same lattice constant).
/// Lambda = a / (2 * sin(theta/2))
pub fn moire_periodicity_twist(a: f64, theta_deg: f64) -> f64 {
    let half_theta = theta_deg.to_radians() / 2.0;
    let sin_val = half_theta.sin().abs();
    if sin_val < 1e-15 {
        return f64::INFINITY;
    }
    a / (2.0 * sin_val)
}

/// Generate the reciprocal lattice G-vectors for a given lattice type.
fn reciprocal_g_vectors(lattice_type: LatticeType, a: f64) -> Vec<Vector2<f64>> {
    match lattice_type {
        LatticeType::Square => {
            let g = 2.0 * PI / a;
            vec![
                Vector2::new(g, 0.0),
                Vector2::new(0.0, g),
                Vector2::new(-g, 0.0),
                Vector2::new(0.0, -g),
            ]
        }
        LatticeType::Hexagonal => {
            // 6 vectors at 60-degree intervals, magnitude 4*pi / (a*sqrt(3))
            let mag = 4.0 * PI / (a * 3.0_f64.sqrt());
            (0..6)
                .map(|i| {
                    let angle = i as f64 * PI / 3.0;
                    Vector2::new(mag * angle.cos(), mag * angle.sin())
                })
                .collect()
        }
    }
}

/// Apply a rotation to a set of G-vectors.
fn rotate_g_vectors(gs: &[Vector2<f64>], angle_rad: f64) -> Vec<Vector2<f64>> {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    gs.iter()
        .map(|g| Vector2::new(g.x * c - g.y * s, g.x * s + g.y * c))
        .collect()
}

/// Compute the moire superlattice pattern via plane-wave superposition.
pub fn compute_moire(config: &MoireConfig) -> MoireResult {
    let n = config.resolution;
    let extent = config.physical_extent;
    let mismatch_percent = ((config.overlayer_a - config.substrate_a) / config.substrate_a).abs() * 100.0;

    // Estimate moire period
    let moire_period = if config.twist_angle_deg.abs() < 0.01 {
        moire_periodicity_1d(config.substrate_a, config.overlayer_a)
    } else if (config.substrate_a - config.overlayer_a).abs() < 0.01 {
        moire_periodicity_twist(config.substrate_a, config.twist_angle_deg)
    } else {
        // Combined effect: approximate using the smaller period
        let p1 = moire_periodicity_1d(config.substrate_a, config.overlayer_a);
        let p2 = moire_periodicity_twist(
            (config.substrate_a + config.overlayer_a) / 2.0,
            config.twist_angle_deg,
        );
        p1.min(p2)
    };

    // Generate G-vectors for substrate
    let g_sub = reciprocal_g_vectors(config.substrate_lattice_type, config.substrate_a);

    // Generate G-vectors for overlayer, then rotate
    let g_over_unrotated = reciprocal_g_vectors(config.overlayer_lattice_type, config.overlayer_a);
    let g_over = rotate_g_vectors(&g_over_unrotated, config.twist_angle_deg.to_radians());

    // Compute pattern on grid
    let mut pattern = vec![0.0_f64; n * n];
    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;

    for iy in 0..n {
        let y = (iy as f64 / (n - 1).max(1) as f64 - 0.5) * extent;
        for ix in 0..n {
            let x = (ix as f64 / (n - 1).max(1) as f64 - 0.5) * extent;
            let r = Vector2::new(x, y);

            // Substrate potential: sum of cos(G . r)
            let v_sub: f64 = g_sub.iter().map(|g| g.dot(&r).cos()).sum();
            // Overlayer potential: sum of cos(G . r)
            let v_over: f64 = g_over.iter().map(|g| g.dot(&r).cos()).sum();

            let val = v_sub * v_over;
            pattern[iy * n + ix] = val;

            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }
    }

    // Normalize to [0, 1]
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

    MoireResult {
        pattern,
        resolution: n,
        physical_extent: extent,
        moire_period,
        mismatch_percent,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_moire_periodicity_1d() {
        // FeTe (3.82) vs Sb2Te3 (4.264): period = 3.82 * 4.264 / |3.82 - 4.264|
        let p = moire_periodicity_1d(3.82, 4.264);
        let expected = 3.82 * 4.264 / (4.264 - 3.82);
        assert!((p - expected).abs() < 1e-10);
    }

    #[test]
    fn test_moire_periodicity_1d_equal() {
        let p = moire_periodicity_1d(4.0, 4.0);
        assert!(p.is_infinite());
    }

    #[test]
    fn test_moire_periodicity_twist() {
        let p = moire_periodicity_twist(4.0, 10.0);
        let expected = 4.0 / (2.0 * (5.0_f64.to_radians()).sin());
        assert!((p - expected).abs() < 1e-10);
    }

    #[test]
    fn test_moire_periodicity_twist_zero() {
        let p = moire_periodicity_twist(4.0, 0.0);
        assert!(p.is_infinite());
    }

    #[test]
    fn test_compute_moire_dimensions() {
        let config = MoireConfig {
            substrate_a: 3.82,
            substrate_lattice_type: LatticeType::Square,
            overlayer_a: 4.264,
            overlayer_lattice_type: LatticeType::Hexagonal,
            twist_angle_deg: 0.0,
            resolution: 64,
            physical_extent: 100.0,
        };
        let result = compute_moire(&config);
        assert_eq!(result.pattern.len(), 64 * 64);
        assert_eq!(result.resolution, 64);
    }

    #[test]
    fn test_compute_moire_normalization() {
        let config = MoireConfig {
            substrate_a: 3.82,
            substrate_lattice_type: LatticeType::Square,
            overlayer_a: 4.264,
            overlayer_lattice_type: LatticeType::Hexagonal,
            twist_angle_deg: 0.0,
            resolution: 32,
            physical_extent: 80.0,
        };
        let result = compute_moire(&config);
        let min_val = result.pattern.iter().cloned().fold(f64::MAX, f64::min);
        let max_val = result.pattern.iter().cloned().fold(f64::MIN, f64::max);
        assert!(min_val >= -1e-10);
        assert!(max_val <= 1.0 + 1e-10);
    }

    #[test]
    fn test_mismatch_percent() {
        let config = MoireConfig {
            substrate_a: 3.82,
            substrate_lattice_type: LatticeType::Square,
            overlayer_a: 4.264,
            overlayer_lattice_type: LatticeType::Hexagonal,
            twist_angle_deg: 0.0,
            resolution: 16,
            physical_extent: 50.0,
        };
        let result = compute_moire(&config);
        let expected = ((4.264 - 3.82) / 3.82) * 100.0;
        assert!((result.mismatch_percent - expected).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocal_g_square_count() {
        let gs = reciprocal_g_vectors(LatticeType::Square, 3.82);
        assert_eq!(gs.len(), 4);
    }

    #[test]
    fn test_reciprocal_g_hex_count() {
        let gs = reciprocal_g_vectors(LatticeType::Hexagonal, 4.264);
        assert_eq!(gs.len(), 6);
    }
}
