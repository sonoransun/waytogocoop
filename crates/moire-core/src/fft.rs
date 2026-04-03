use num_complex::Complex64;
use rustfft::FftPlanner;

/// Compute a 2D FFT of a real-valued NxN grid and return the log-scaled,
/// fftshift-ed power spectrum as an NxN vector of f64 values normalized to [0, 1].
pub fn compute_fft_2d(data: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(data.len(), n * n, "data length must be n*n");

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

    shifted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_output_size() {
        let data = vec![0.0_f64; 16];
        let result = compute_fft_2d(&data, 4);
        assert_eq!(result.len(), 16);
    }

    #[test]
    fn test_fft_uniform_input() {
        // Uniform input should have all energy at DC
        let n = 8;
        let data = vec![1.0_f64; n * n];
        let result = compute_fft_2d(&data, n);
        // DC component (center after shift) should be the maximum
        let center = n / 2 * n + n / 2;
        let max_val = result.iter().copied().fold(f64::MIN, f64::max);
        assert!((result[center] - max_val).abs() < 1e-10);
    }

    #[test]
    fn test_fft_normalization() {
        let n = 8;
        let data: Vec<f64> = (0..n * n).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = compute_fft_2d(&data, n);
        let min_val = result.iter().copied().fold(f64::MAX, f64::min);
        let max_val = result.iter().copied().fold(f64::MIN, f64::max);
        assert!(min_val >= -1e-10);
        assert!(max_val <= 1.0 + 1e-10);
    }

    #[test]
    #[should_panic(expected = "data length must be n*n")]
    fn test_fft_wrong_size() {
        let data = vec![0.0_f64; 10];
        compute_fft_2d(&data, 4);
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
        let result = compute_fft_2d(&data, n);
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
        let result = compute_fft_2d(&input, 8);
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
        let result = compute_fft_2d(&input, 8);
        // Center pixel (DC) should be the maximum
        let center = result[4 * 8 + 4]; // n/2 * n + n/2 for n=8
        let max_val = result.iter().copied().fold(f64::MIN, f64::max);
        assert!((center - max_val).abs() < 1e-10);
    }
}
