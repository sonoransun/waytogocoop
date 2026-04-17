//! 2D FFT + peak detection for moire patterns.
//!
//! Uses rustfft for the forward transform, applies `log(1+|F|²)` scaling and
//! fftshift so DC lands in the center, then normalizes to `[0, 1]` for
//! colormap display. Peak detection extracts local maxima above a threshold.
//! Mirrors `src/waytogocoop/computation/fourier.py`.

use num_complex::Complex64;
use rustfft::FftPlanner;

/// Compute a 2D FFT of a real-valued NxN grid and return the log-scaled,
/// fftshift-ed power spectrum as an NxN vector of f64 values normalized to [0, 1].
pub fn compute_fft_2d(data: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if n == 0 {
        return Err("n must be > 0".to_string());
    }
    if data.len() != n * n {
        return Err(format!("data length {} != n*n ({})", data.len(), n * n));
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    // Convert to complex and perform row-wise FFT
    let mut buf: Vec<Complex64> = data.iter().map(|&v| Complex64::new(v, 0.0)).collect();

    // Row-wise FFT
    for row in 0..n {
        let start = row * n;
        let end = start + n;
        fft.process(&mut buf[start..end]);
    }

    // Column-wise FFT: extract columns, FFT, put back
    let mut col_buf = vec![Complex64::new(0.0, 0.0); n];
    for col in 0..n {
        for row in 0..n {
            col_buf[row] = buf[row * n + col];
        }
        fft.process(&mut col_buf);
        for row in 0..n {
            buf[row * n + col] = col_buf[row];
        }
    }

    // Compute power spectrum (magnitude squared)
    let mut power: Vec<f64> = buf.iter().map(|c| c.norm_sqr()).collect();

    // Log-scale: log(1 + value) to handle zeros
    for v in power.iter_mut() {
        *v = (*v + 1.0).ln();
    }

    // FFT shift: swap quadrants so DC is in center
    let mut shifted = vec![0.0_f64; n * n];
    let half = n / 2;
    for iy in 0..n {
        for ix in 0..n {
            let sy = (iy + half) % n;
            let sx = (ix + half) % n;
            shifted[sy * n + sx] = power[iy * n + ix];
        }
    }

    // Normalize to [0, 1]
    let min_val = shifted.iter().copied().fold(f64::MAX, f64::min);
    let max_val = shifted.iter().copied().fold(f64::MIN, f64::max);
    let range = max_val - min_val;
    if range > 1e-15 {
        for v in shifted.iter_mut() {
            *v = (*v - min_val) / range;
        }
    } else {
        for v in shifted.iter_mut() {
            *v = 0.5;
        }
    }

    Ok(shifted)
}

/// A peak found in a 2D FFT power spectrum.
#[derive(Debug, Clone)]
pub struct FftPeak {
    pub kx: f64,
    pub ky: f64,
    pub amplitude: f64,
}

/// Compute shifted FFT frequency axis, matching numpy's fftshift(fftfreq(n, dx)) * 2*pi.
pub fn fft_frequencies(n: usize, dx: f64) -> Vec<f64> {
    // fftfreq: [0, 1, 2, ..., n/2-1, -n/2, -n/2+1, ..., -1] / (n*dx)
    // then fftshift: [-n/2, ..., -1, 0, 1, ..., n/2-1] / (n*dx)
    // then * 2*pi for angular frequency
    let factor = 2.0 * std::f64::consts::PI / (n as f64 * dx);
    let half = n as i64 / 2;
    (0..n)
        .map(|i| {
            let shifted = i as i64 - half;
            shifted as f64 * factor
        })
        .collect()
}

/// Identify peaks in a 2D FFT power spectrum (local maxima above threshold, excluding DC).
pub fn identify_peaks(
    power_spectrum: &[f64],
    n: usize,
    kx: &[f64],
    ky: &[f64],
    threshold_fraction: f64,
) -> Vec<FftPeak> {
    if power_spectrum.len() != n * n || kx.len() != n || ky.len() != n {
        return Vec::new();
    }

    // Find global max
    let global_max = power_spectrum.iter().copied().fold(0.0_f64, f64::max);
    if global_max < 1e-30 {
        return Vec::new();
    }

    let threshold = threshold_fraction * global_max;
    let neighbourhood_size = (n / 20).max(5);
    let margin = (neighbourhood_size / 2).max(2);
    let half = n / 2;

    let mut peaks = Vec::new();

    for row in 0..n {
        for col in 0..n {
            // Exclude DC component: pixels within margin of center
            let dr = (row as i64 - half as i64).unsigned_abs() as usize;
            let dc = (col as i64 - half as i64).unsigned_abs() as usize;
            if dr < margin && dc < margin {
                continue;
            }

            let val = power_spectrum[row * n + col];
            if val < threshold {
                continue;
            }

            // Check if this pixel is the maximum in its neighbourhood
            let half_nb = neighbourhood_size / 2;
            let r_start = if row >= half_nb { row - half_nb } else { 0 };
            let r_end = (row + half_nb + 1).min(n);
            let c_start = if col >= half_nb { col - half_nb } else { 0 };
            let c_end = (col + half_nb + 1).min(n);

            let mut is_max = true;
            'outer: for nr in r_start..r_end {
                for nc in c_start..c_end {
                    if nr == row && nc == col {
                        continue;
                    }
                    if power_spectrum[nr * n + nc] > val {
                        is_max = false;
                        break 'outer;
                    }
                }
            }

            if is_max {
                peaks.push(FftPeak {
                    kx: kx[col],
                    ky: ky[row],
                    amplitude: val,
                });
            }
        }
    }

    // Sort by amplitude descending
    peaks.sort_by(|a, b| b.amplitude.partial_cmp(&a.amplitude).unwrap_or(std::cmp::Ordering::Equal));

    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_output_size() {
        let data = vec![0.0_f64; 16];
        let result = compute_fft_2d(&data, 4).unwrap();
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_fft_uniform_input() {
        // Uniform input should have all energy at DC
        let n = 8;
        let data = vec![1.0_f64; n * n];
        let result = compute_fft_2d(&data, n).unwrap();
        // DC component (center after shift) should be the maximum
        let center = n / 2 * n + n / 2;
        let max_val = result.iter().copied().fold(f64::MIN, f64::max);
        assert!((result[center] - max_val).abs() < 1e-10);
    }

    #[test]
    fn test_fft_normalization() {
        let n = 8;
        let data: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = compute_fft_2d(&data, n).unwrap();
        let min_val = result.iter().copied().fold(f64::MAX, f64::min);
        let max_val = result.iter().copied().fold(f64::MIN, f64::max);
        assert!(min_val >= -1e-10);
        assert!(max_val <= 1.0 + 1e-10);
    }

    #[test]
    fn test_fft_wrong_size() {
        let data = vec![0.0_f64; 10];
        let result = compute_fft_2d(&data, 4);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("data length"));
    }

    #[test]
    fn test_fft_sinusoidal() {
        // A sinusoidal pattern should produce peaks at the corresponding frequency
        let n = 16;
        let freq = 2.0;
        let data: Vec<f64> = (0..n * n)
            .map(|idx| {
                let x = idx % n;
                (2.0 * std::f64::consts::PI * freq * x as f64 / n as f64).cos()
            })
            .collect();
        let result = compute_fft_2d(&data, n).unwrap();
        // Result should not be all identical (i.e., there are spectral features)
        let min_val = result.iter().copied().fold(f64::MAX, f64::min);
        let max_val = result.iter().copied().fold(f64::MIN, f64::max);
        assert!(max_val - min_val > 0.01);
    }

    #[test]
    fn test_fft_all_zeros() {
        // All zeros: FFT gives all zeros, log(0+1) = 0 for all,
        // range = 0, so normalization falls back to all 0.5
        let input = vec![0.0; 64];
        let result = compute_fft_2d(&input, 8).unwrap();
        // Should not panic. All values should be the same (fallback normalization)
        let first = result[0];
        for &v in &result {
            assert!((v - first).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fft_dc_only() {
        // Constant nonzero input: all energy at DC
        let input = vec![5.0; 64];
        let result = compute_fft_2d(&input, 8).unwrap();
        // Center pixel (DC) should be the maximum
        let center = result[4 * 8 + 4]; // n/2 * n + n/2 for n=8
        let max_val = result.iter().copied().fold(f64::MIN, f64::max);
        assert!((center - max_val).abs() < 1e-10);
    }

    #[test]
    fn test_fft_n_zero_err() {
        assert!(compute_fft_2d(&[], 0).is_err());
    }

    #[test]
    fn test_fft_mismatched_size_err() {
        assert!(compute_fft_2d(&[1.0, 2.0, 3.0], 2).is_err()); // len 3 != 2*2
    }

    #[test]
    fn test_fft_frequencies_length() {
        let freqs = fft_frequencies(64, 1.0);
        assert_eq!(freqs.len(), 64);
    }

    #[test]
    fn test_fft_frequencies_symmetry() {
        let freqs = fft_frequencies(64, 1.0);
        // Should be approximately symmetric around 0
        assert!(freqs[0] < 0.0); // negative frequencies first after shift
        assert!(freqs[63] > 0.0);
    }

    #[test]
    fn test_identify_peaks_empty_on_zero() {
        let n = 16;
        let data = vec![0.0; n * n]; // all-zero spectrum
        let kx = fft_frequencies(n, 1.0);
        let ky = fft_frequencies(n, 1.0);
        let peaks = identify_peaks(&data, n, &kx, &ky, 0.5);
        // global_max < 1e-30 → early return with empty vec
        assert!(peaks.is_empty());
    }

    #[test]
    fn test_identify_peaks_sorted_by_amplitude() {
        // Create a spectrum with known peaks
        let n = 32;
        let mut data = vec![0.0; n * n];
        // Place peaks away from center
        data[5 * n + 5] = 10.0;
        data[25 * n + 25] = 5.0;
        let kx = fft_frequencies(n, 1.0);
        let ky = fft_frequencies(n, 1.0);
        let peaks = identify_peaks(&data, n, &kx, &ky, 0.1);
        if peaks.len() >= 2 {
            assert!(peaks[0].amplitude >= peaks[1].amplitude);
        }
    }
}
