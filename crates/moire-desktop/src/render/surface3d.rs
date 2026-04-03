use egui::{Color32, ColorImage};

/// Camera state for the 3D surface view.
#[derive(Debug, Clone, Copy)]
pub struct Camera3D {
    /// Horizontal rotation angle (radians).
    pub yaw: f32,
    /// Vertical rotation angle (radians).
    pub pitch: f32,
    /// Distance from the origin.
    pub distance: f32,
}

impl Default for Camera3D {
    fn default() -> Self {
        Self {
            yaw: 0.8,
            pitch: 0.6,
            distance: 2.5,
        }
    }
}

/// Render a 2D heightmap as a 3D perspective-projected surface.
///
/// `data` is row-major `n x n` values in [0, 1].
/// `bg_color` is the background fill color for unpainted pixels.
/// Returns a `ColorImage` of `width x height` pixels.
pub fn render_surface_3d(
    data: &[f64],
    n: usize,
    width: usize,
    height: usize,
    camera: &Camera3D,
    colormap: fn(f64) -> [u8; 4],
) -> ColorImage {
    render_surface_3d_with_bg(data, n, width, height, camera, colormap, Color32::from_rgb(30, 30, 35))
}

/// Like `render_surface_3d` but with an explicit background color.
pub fn render_surface_3d_with_bg(
    data: &[f64],
    n: usize,
    width: usize,
    height: usize,
    camera: &Camera3D,
    colormap: fn(f64) -> [u8; 4],
    bg_color: Color32,
) -> ColorImage {
    let mut pixels = vec![bg_color; width * height];
    let mut zbuf = vec![f32::MAX; width * height];

    // Subsample: target ~64x64 mesh
    let step = (n / 64).max(1);
    let mesh_n = (n + step - 1) / step;

    // Build vertex grid: x,y in [-1,1], z = height * scale
    let height_scale = 0.3_f32;
    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(mesh_n * mesh_n);
    let mut values: Vec<f64> = Vec::with_capacity(mesh_n * mesh_n);

    for iy in 0..mesh_n {
        let sy = (iy * step).min(n - 1);
        let y = (sy as f32 / n as f32) * 2.0 - 1.0;
        for ix in 0..mesh_n {
            let sx = (ix * step).min(n - 1);
            let x = (sx as f32 / n as f32) * 2.0 - 1.0;
            let val = data[sy * n + sx];
            let z = val as f32 * height_scale;
            verts.push([x, y, z]);
            values.push(val);
        }
    }

    // Precompute rotation
    let cy = camera.yaw.cos();
    let sy_r = camera.yaw.sin();
    let cp = camera.pitch.cos();
    let sp = camera.pitch.sin();
    let dist = camera.distance;
    let focal = 2.0_f32;

    let transform = |p: [f32; 3]| -> [f32; 3] {
        // Rotate around Z by yaw
        let x1 = p[0] * cy - p[1] * sy_r;
        let y1 = p[0] * sy_r + p[1] * cy;
        let z1 = p[2];
        // Rotate around X by pitch
        let x2 = x1;
        let y2 = y1 * cp - z1 * sp;
        let z2 = y1 * sp + z1 * cp;
        // Translate back by distance
        let z3 = z2 + dist;
        // Perspective project
        if z3 < 0.01 {
            return [f32::NAN, f32::NAN, f32::NAN];
        }
        let sx = focal * x2 / z3;
        let sy = focal * y2 / z3;
        // Map to screen coordinates
        let px = (sx + 1.0) * 0.5 * width as f32;
        let py = (1.0 - (sy + 1.0) * 0.5) * height as f32;
        [px, py, z3]
    };

    // Transform all vertices
    let screen_verts: Vec<[f32; 3]> = verts.iter().map(|v| transform(*v)).collect();

    // Light direction (normalized)
    let light = normalize([0.3, 0.5, 0.8]);

    // Rasterize quads as triangle pairs
    for iy in 0..mesh_n - 1 {
        for ix in 0..mesh_n - 1 {
            let i00 = iy * mesh_n + ix;
            let i10 = iy * mesh_n + ix + 1;
            let i01 = (iy + 1) * mesh_n + ix;
            let i11 = (iy + 1) * mesh_n + ix + 1;

            let avg_val = (values[i00] + values[i10] + values[i01] + values[i11]) * 0.25;
            let [r, g, b, _] = colormap(avg_val);

            // Triangle 1: i00, i10, i01
            let v0 = screen_verts[i00];
            let v1 = screen_verts[i10];
            let v2 = screen_verts[i01];
            if !v0[0].is_nan() && !v1[0].is_nan() && !v2[0].is_nan() {
                let normal = face_normal(verts[i00], verts[i10], verts[i01]);
                let shade = shade_factor(normal, light);
                let c = shade_color(r, g, b, shade);
                rasterize_triangle(v0, v1, v2, c, &mut pixels, &mut zbuf, width, height);
            }

            // Triangle 2: i10, i11, i01
            let v0 = screen_verts[i10];
            let v1 = screen_verts[i11];
            let v2 = screen_verts[i01];
            if !v0[0].is_nan() && !v1[0].is_nan() && !v2[0].is_nan() {
                let normal = face_normal(verts[i10], verts[i11], verts[i01]);
                let shade = shade_factor(normal, light);
                let c = shade_color(r, g, b, shade);
                rasterize_triangle(v0, v1, v2, c, &mut pixels, &mut zbuf, width, height);
            }
        }
    }

    ColorImage {
        size: [width, height],
        pixels,
    }
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [0.0, 0.0, 1.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    normalize([
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ])
}

fn shade_factor(normal: [f32; 3], light: [f32; 3]) -> f32 {
    let dot = normal[0] * light[0] + normal[1] * light[1] + normal[2] * light[2];
    let ambient = 0.25_f32;
    let diffuse = dot.abs() * 0.75;
    (ambient + diffuse).min(1.0)
}

fn shade_color(r: u8, g: u8, b: u8, shade: f32) -> Color32 {
    Color32::from_rgb(
        (r as f32 * shade) as u8,
        (g as f32 * shade) as u8,
        (b as f32 * shade) as u8,
    )
}

fn rasterize_triangle(
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    color: Color32,
    pixels: &mut [Color32],
    zbuf: &mut [f32],
    width: usize,
    height: usize,
) {
    // Bounding box
    let min_x = v0[0].min(v1[0]).min(v2[0]).max(0.0) as i32;
    let max_x = v0[0].max(v1[0]).max(v2[0]).min(width as f32 - 1.0) as i32;
    let min_y = v0[1].min(v1[1]).min(v2[1]).max(0.0) as i32;
    let max_y = v0[1].max(v1[1]).max(v2[1]).min(height as f32 - 1.0) as i32;

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_fn(v0, v1, v2);
    if area.abs() < 0.001 {
        return; // degenerate triangle
    }
    let inv_area = 1.0 / area;

    for py in min_y..=max_y {
        for px in min_x..=max_x {
            let p = [px as f32 + 0.5, py as f32 + 0.5];
            let w0 = edge_fn_2d(v1, v2, p) * inv_area;
            let w1 = edge_fn_2d(v2, v0, p) * inv_area;
            let w2 = edge_fn_2d(v0, v1, p) * inv_area;

            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                let z = w0 * v0[2] + w1 * v1[2] + w2 * v2[2];
                let idx = py as usize * width + px as usize;
                if z < zbuf[idx] {
                    zbuf[idx] = z;
                    pixels[idx] = color;
                }
            }
        }
    }
}

fn edge_fn(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    (c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0])
}

fn edge_fn_2d(a: [f32; 3], b: [f32; 3], p: [f32; 2]) -> f32 {
    (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camera_default() {
        let cam = Camera3D::default();
        assert!(cam.yaw > 0.0);
        assert!(cam.pitch > 0.0);
        assert!(cam.distance > 1.0);
    }

    #[test]
    fn test_render_produces_correct_size() {
        let data = vec![0.5; 16 * 16];
        let img = render_surface_3d(&data, 16, 64, 64, &Camera3D::default(), |t| {
            let v = (t * 255.0) as u8;
            [v, v, v, 255]
        });
        assert_eq!(img.size, [64, 64]);
        assert_eq!(img.pixels.len(), 64 * 64);
    }

    #[test]
    fn test_render_not_all_background() {
        let mut data = vec![0.0; 32 * 32];
        // Create a visible height variation
        for i in 0..32 {
            for j in 0..32 {
                data[i * 32 + j] = (i as f64 + j as f64) / 62.0;
            }
        }
        let bg = Color32::from_rgb(30, 30, 35);
        let img = render_surface_3d(&data, 32, 128, 128, &Camera3D::default(), |t| {
            let v = (t * 255.0) as u8;
            [v, v, v, 255]
        });
        let non_bg = img.pixels.iter().filter(|&&p| p != bg).count();
        assert!(non_bg > 100, "Expected rendered pixels, got only {} non-background", non_bg);
    }

    #[test]
    fn test_normalize() {
        let n = normalize([3.0, 4.0, 0.0]);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_shade_factor_range() {
        let n = normalize([0.0, 0.0, 1.0]);
        let l = normalize([0.0, 0.0, 1.0]);
        let s = shade_factor(n, l);
        assert!(s >= 0.0 && s <= 1.0);
    }
}
