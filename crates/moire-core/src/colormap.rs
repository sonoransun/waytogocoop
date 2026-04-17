//! Colormap palettes for scalar-field visualization.
//!
//! Convention (must stay in sync with Python `figure_factory.py`):
//! - `viridis`  — unsigned scalar fields (moire pattern, generic surface).
//! - `coolwarm` — diverging / signed fields (gap modulation, combined gap).
//! - `inferno`  — FFT power spectra (log-scaled, normalized).
//!
//! Each palette maps a value `t` in `[0, 1]` to an RGBA quadruplet.

/// Interpolate between colormap control points.
/// Each control point is (t, r, g, b) where t is in [0, 1] and r,g,b are in [0, 255].
fn interpolate_colormap(t: f64, points: &[(f64, f64, f64, f64)]) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);

    // Find the two bounding control points
    if t <= points[0].0 {
        let p = &points[0];
        return [p.1 as u8, p.2 as u8, p.3 as u8, 255];
    }
    if t >= points[points.len() - 1].0 {
        let p = &points[points.len() - 1];
        return [p.1 as u8, p.2 as u8, p.3 as u8, 255];
    }

    for i in 0..points.len() - 1 {
        if t >= points[i].0 && t <= points[i + 1].0 {
            let t0 = points[i].0;
            let t1 = points[i + 1].0;
            let frac = (t - t0) / (t1 - t0);
            let r = points[i].1 + frac * (points[i + 1].1 - points[i].1);
            let g = points[i].2 + frac * (points[i + 1].2 - points[i].2);
            let b = points[i].3 + frac * (points[i + 1].3 - points[i].3);
            return [r as u8, g as u8, b as u8, 255];
        }
    }

    // Fallback
    [0, 0, 0, 255]
}

// Viridis colormap control points (32 samples from matplotlib's viridis)
static VIRIDIS: &[(f64, f64, f64, f64)] = &[
    (0.000, 68.0, 1.0, 84.0),
    (0.033, 72.0, 15.0, 104.0),
    (0.067, 72.0, 29.0, 119.0),
    (0.100, 70.0, 43.0, 131.0),
    (0.133, 65.0, 55.0, 140.0),
    (0.167, 59.0, 66.0, 146.0),
    (0.200, 53.0, 77.0, 150.0),
    (0.233, 47.0, 86.0, 152.0),
    (0.267, 42.0, 96.0, 153.0),
    (0.300, 37.0, 105.0, 153.0),
    (0.333, 33.0, 114.0, 151.0),
    (0.367, 30.0, 122.0, 148.0),
    (0.400, 28.0, 130.0, 144.0),
    (0.433, 27.0, 138.0, 139.0),
    (0.467, 29.0, 146.0, 132.0),
    (0.500, 34.0, 153.0, 124.0),
    (0.533, 44.0, 160.0, 114.0),
    (0.567, 58.0, 167.0, 103.0),
    (0.600, 76.0, 173.0, 91.0),
    (0.633, 96.0, 179.0, 78.0),
    (0.667, 119.0, 184.0, 65.0),
    (0.700, 143.0, 189.0, 52.0),
    (0.733, 167.0, 192.0, 40.0),
    (0.767, 192.0, 195.0, 31.0),
    (0.800, 215.0, 196.0, 28.0),
    (0.833, 234.0, 195.0, 33.0),
    (0.867, 248.0, 192.0, 45.0),
    (0.900, 253.0, 202.0, 53.0),
    (0.933, 253.0, 216.0, 62.0),
    (0.967, 251.0, 231.0, 74.0),
    (1.000, 253.0, 231.0, 37.0),
];

// Inferno colormap control points (32 samples from matplotlib's inferno)
static INFERNO: &[(f64, f64, f64, f64)] = &[
    (0.000, 0.0, 0.0, 4.0),
    (0.033, 7.0, 4.0, 26.0),
    (0.067, 18.0, 9.0, 52.0),
    (0.100, 32.0, 12.0, 74.0),
    (0.133, 48.0, 11.0, 91.0),
    (0.167, 64.0, 10.0, 103.0),
    (0.200, 80.0, 10.0, 110.0),
    (0.233, 96.0, 14.0, 112.0),
    (0.267, 112.0, 20.0, 109.0),
    (0.300, 127.0, 29.0, 103.0),
    (0.333, 141.0, 39.0, 94.0),
    (0.367, 154.0, 49.0, 83.0),
    (0.400, 166.0, 60.0, 72.0),
    (0.433, 177.0, 72.0, 60.0),
    (0.467, 187.0, 84.0, 49.0),
    (0.500, 196.0, 97.0, 38.0),
    (0.533, 204.0, 111.0, 28.0),
    (0.567, 211.0, 125.0, 19.0),
    (0.600, 218.0, 139.0, 12.0),
    (0.633, 223.0, 154.0, 8.0),
    (0.667, 228.0, 170.0, 10.0),
    (0.700, 231.0, 186.0, 21.0),
    (0.733, 233.0, 201.0, 40.0),
    (0.767, 234.0, 217.0, 66.0),
    (0.800, 235.0, 232.0, 96.0),
    (0.833, 237.0, 242.0, 125.0),
    (0.867, 240.0, 248.0, 154.0),
    (0.900, 244.0, 252.0, 180.0),
    (0.933, 248.0, 254.0, 205.0),
    (0.967, 252.0, 254.0, 229.0),
    (1.000, 252.0, 255.0, 252.0),
];

// Cool-warm (diverging blue-red) colormap control points
static COOLWARM: &[(f64, f64, f64, f64)] = &[
    (0.000, 59.0, 76.0, 192.0),
    (0.067, 74.0, 96.0, 209.0),
    (0.133, 94.0, 118.0, 222.0),
    (0.200, 117.0, 140.0, 232.0),
    (0.267, 142.0, 162.0, 239.0),
    (0.333, 167.0, 183.0, 243.0),
    (0.400, 192.0, 203.0, 245.0),
    (0.467, 216.0, 220.0, 241.0),
    (0.500, 230.0, 230.0, 230.0),
    (0.533, 239.0, 216.0, 211.0),
    (0.600, 243.0, 196.0, 185.0),
    (0.667, 241.0, 172.0, 156.0),
    (0.733, 234.0, 146.0, 127.0),
    (0.800, 222.0, 118.0, 100.0),
    (0.867, 206.0, 88.0, 74.0),
    (0.933, 186.0, 56.0, 51.0),
    (1.000, 180.0, 4.0, 38.0),
];

/// Map a value t in [0, 1] to viridis RGBA color.
pub fn viridis(t: f64) -> [u8; 4] {
    interpolate_colormap(t, VIRIDIS)
}

/// Map a value t in [0, 1] to inferno RGBA color.
pub fn inferno(t: f64) -> [u8; 4] {
    interpolate_colormap(t, INFERNO)
}

/// Map a value t in [0, 1] to cool-warm (diverging) RGBA color.
pub fn coolwarm(t: f64) -> [u8; 4] {
    interpolate_colormap(t, COOLWARM)
}

/// Apply a colormap function to a slice of [0, 1] values, producing an RGBA pixel buffer.
pub fn apply_colormap(data: &[f64], colormap: fn(f64) -> [u8; 4]) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(data.len() * 4);
    for &val in data {
        let [r, g, b, a] = colormap(val);
        pixels.push(r);
        pixels.push(g);
        pixels.push(b);
        pixels.push(a);
    }
    pixels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_viridis_endpoints() {
        let c0 = viridis(0.0);
        assert_eq!(c0[3], 255); // fully opaque
        let c1 = viridis(1.0);
        assert_eq!(c1[3], 255);
        // Viridis starts dark purple, ends yellow-green
        // c0 should be dark-ish, c1 should be bright
        assert!(c0[0] < 100);
        assert!(c1[0] > 200);
    }

    #[test]
    fn test_inferno_endpoints() {
        let c0 = inferno(0.0);
        // Inferno starts nearly black
        assert!(c0[0] < 10);
        assert!(c0[1] < 10);
        let c1 = inferno(1.0);
        // Inferno ends nearly white
        assert!(c1[0] > 240);
        assert!(c1[1] > 240);
    }

    #[test]
    fn test_coolwarm_midpoint() {
        let mid = coolwarm(0.5);
        // Should be near gray/white at midpoint
        assert!(mid[0] > 200);
        assert!(mid[1] > 200);
        assert!(mid[2] > 200);
    }

    #[test]
    fn test_clamping() {
        // Values outside [0, 1] should be clamped
        let below = viridis(-0.5);
        let at_zero = viridis(0.0);
        assert_eq!(below, at_zero);

        let above = viridis(1.5);
        let at_one = viridis(1.0);
        assert_eq!(above, at_one);
    }

    #[test]
    fn test_apply_colormap_length() {
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let pixels = apply_colormap(&data, viridis);
        assert_eq!(pixels.len(), 5 * 4); // RGBA for each pixel
    }

    #[test]
    fn test_apply_colormap_all_opaque() {
        let data = vec![0.0, 0.5, 1.0];
        let pixels = apply_colormap(&data, inferno);
        // Every 4th byte (alpha) should be 255
        for i in 0..3 {
            assert_eq!(pixels[i * 4 + 3], 255);
        }
    }
}
