use crate::moire::MoireResult;
use std::f64::consts::PI;

/// Configuration for Cooper-pair density modulation computation.
#[derive(Debug, Clone)]
pub struct DensityConfig {
    /// First superconducting gap in meV.
    pub delta_1: f64,
    /// Second superconducting gap in meV.
    pub delta_2: f64,
    /// Modulation amplitude as a fraction.
    pub modulation_amplitude: f64,
    /// Phase shift in radians.
    pub phase_shift: f64,
}

impl Default for DensityConfig {
    fn default() -> Self {
        Self {
            delta_1: 2.58,
            delta_2: 3.60,
            modulation_amplitude: 0.15,
            phase_shift: PI,
        }
    }
}

/// Result of the density modulation computation.
#[derive(Debug, Clone)]
pub struct DensityResult {
    /// Spatially modulated superconducting gap field, same grid as the moire pattern.
    pub gap_field: Vec<f64>,
    pub resolution: usize,
}

/// Compute the Cooper-pair density modulation from a moire pattern.
///
/// gap_field[i] = delta_avg + delta_amplitude * (moire.pattern[i] - 0.5) * 2.0
///
/// where delta_avg = (delta_1 + delta_2) / 2,
///       delta_amplitude = modulation_amplitude * delta_avg
pub fn compute_density_modulation(
    moire: &MoireResult,
    config: &DensityConfig,
) -> DensityResult {
    let delta_avg = (config.delta_1 + config.delta_2) / 2.0;
    let delta_amplitude = config.modulation_amplitude * delta_avg;

    let gap_field: Vec<f64> = moire
        .pattern
        .iter()
        .map(|&p| delta_avg + delta_amplitude * (p - 0.5) * 2.0)
        .collect();

    DensityResult {
        gap_field,
        resolution: moire.resolution,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_moire() -> MoireResult {
        // 4x4 pattern with known values
        MoireResult {
            pattern: vec![
                0.0, 0.25, 0.5, 0.75,
                0.25, 0.5, 0.75, 1.0,
                0.5, 0.75, 1.0, 0.0,
                0.75, 1.0, 0.0, 0.25,
            ],
            resolution: 4,
            physical_extent: 40.0,
            moire_period: 20.0,
            mismatch_percent: 11.6,
        }
    }

    #[test]
    fn test_default_config() {
        let cfg = DensityConfig::default();
        assert!((cfg.delta_1 - 2.58).abs() < 1e-10);
        assert!((cfg.delta_2 - 3.60).abs() < 1e-10);
        assert!((cfg.modulation_amplitude - 0.15).abs() < 1e-10);
        assert!((cfg.phase_shift - PI).abs() < 1e-10);
    }

    #[test]
    fn test_density_result_size() {
        let moire = make_test_moire();
        let cfg = DensityConfig::default();
        let result = compute_density_modulation(&moire, &cfg);
        assert_eq!(result.gap_field.len(), 16);
        assert_eq!(result.resolution, 4);
    }

    #[test]
    fn test_density_at_midpoint() {
        // When moire pattern = 0.5, gap should equal delta_avg
        let moire = make_test_moire();
        let cfg = DensityConfig::default();
        let result = compute_density_modulation(&moire, &cfg);
        let delta_avg = (cfg.delta_1 + cfg.delta_2) / 2.0;
        // pattern[2] = 0.5 (index 2 in the first row)
        assert!((result.gap_field[2] - delta_avg).abs() < 1e-10);
    }

    #[test]
    fn test_density_range() {
        let moire = make_test_moire();
        let cfg = DensityConfig::default();
        let result = compute_density_modulation(&moire, &cfg);
        let delta_avg = (cfg.delta_1 + cfg.delta_2) / 2.0;
        let delta_amplitude = cfg.modulation_amplitude * delta_avg;

        let min_expected = delta_avg - delta_amplitude;
        let max_expected = delta_avg + delta_amplitude;

        let min_val = result.gap_field.iter().cloned().fold(f64::MAX, f64::min);
        let max_val = result.gap_field.iter().cloned().fold(f64::MIN, f64::max);

        assert!(min_val >= min_expected - 1e-10);
        assert!(max_val <= max_expected + 1e-10);
    }

    #[test]
    fn test_density_symmetry() {
        // pattern=0 and pattern=1 should be equidistant from delta_avg
        let moire = make_test_moire();
        let cfg = DensityConfig::default();
        let result = compute_density_modulation(&moire, &cfg);
        let delta_avg = (cfg.delta_1 + cfg.delta_2) / 2.0;

        // pattern[0] = 0.0, pattern[7] = 1.0
        let dev_low = (result.gap_field[0] - delta_avg).abs();
        let dev_high = (result.gap_field[7] - delta_avg).abs();
        assert!((dev_low - dev_high).abs() < 1e-10);
    }
}
