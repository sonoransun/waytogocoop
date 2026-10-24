//! Software rasterizer for the 3D surface view.
//!
//! CPU-side rasterizer over a subsampled mesh with per-face diffuse shading,
//! perspective projection, and an optional wireframe overlay. World axes and a
//! physical-extent scale bar can be layered on top for orientation.
//!
//! Orientation: the data grid spans the x-y plane (column index along `+x`,
//! row index along `+y`) and the plotted value is the height `z`, which draws
//! **upward**: the eye sits above the plane, so maxima stand up and the blue
//! `+z` axis points up. The GPU backend (`render/gpu`: the same mesh layout
//! viewed through `look_at_rh` with `up = +z`) shares this convention. The
//! camera model is documented on `Projector`.

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

/// World height of a data value of 1.0; the plane spans [-1, 1] in x and y.
const HEIGHT_SCALE: f32 = 0.3;

/// Length of the x and y world axes in plane units (the z axis is 0.6 of it).
const AXIS_LEN: f32 = 1.15;

/// Focal length of the perspective projection: the frame spans `±1/FOCAL` of
/// the depth, a vertical field of view of 2·atan(1/2) ≈ 53°, as on the GPU
/// path.
const FOCAL: f32 = 2.0;

/// Smallest camera-frame depth that still projects; nearer points are
/// dropped.
const NEAR_PLANE: f32 = 0.01;

/// Perspective projection shared by the mesh, the wireframe and the axes.
///
/// World frame: `x` and `y` span the data plane over [-1, 1] (column and row
/// index, so `+y` is the last row), `z` is the height, and larger values draw
/// higher. The camera orbits the origin: `yaw` turns the plane about `z`,
/// `pitch` tilts the view away from a plan view (0) toward the horizon
/// (±π/2), and `distance` is the eye's distance from the origin. In the yawed
/// frame the eye sits at `(0, -d·sin pitch, d·cos pitch)`, above the plane
/// for |pitch| < π/2 and looking at the origin; its forward direction is
/// `(0, sin pitch, -cos pitch)` and screen-up is `(0, cos pitch, sin pitch)`,
/// so a flat plane frames exactly as it did when heights were still drawn
/// downward, while taller points are now nearer to the eye and higher on
/// screen. The GPU path (`render/gpu/camera.rs`, `look_at_rh` with
/// `up = +z`) shares the convention.
pub(crate) struct Projector {
    cos_yaw: f32,
    sin_yaw: f32,
    cos_pitch: f32,
    sin_pitch: f32,
    distance: f32,
    width: f32,
    height: f32,
}

impl Projector {
    pub(crate) fn new(camera: &Camera3D, width: usize, height: usize) -> Self {
        Self {
            cos_yaw: camera.yaw.cos(),
            sin_yaw: camera.yaw.sin(),
            cos_pitch: camera.pitch.cos(),
            sin_pitch: camera.pitch.sin(),
            distance: camera.distance,
            width: width as f32,
            height: height as f32,
        }
    }

    /// Project a world point to `[pixel_x, pixel_y, depth]`; pixel y grows
    /// downward. A point nearer than `NEAR_PLANE` returns NaNs, and the
    /// caller drops every face or line that touches one.
    ///
    /// Near-plane clearance: over the mesh corners (|y| ≤ √2 after the yaw),
    /// heights in [0, `HEIGHT_SCALE`] and the axis tips, the depth is at least
    /// `distance - √(2 + HEIGHT_SCALE²) ≈ distance - 1.446`, so every camera
    /// the app can reach (pitch clamped to ±1.4) is safe once `distance ≥ 1.5`.
    /// At the zoom limit, `distance` 1.0, the tallest vertices of the nearest
    /// corner fall behind the near plane for pitch ≳ 0.55 (the flat corner
    /// from pitch ≳ 0.78, as before the sign fix). They would project far
    /// below the frame anyway, so the dropped faces are off-screen.
    pub(crate) fn project(&self, p: [f32; 3]) -> [f32; 3] {
        // Yaw: turn the plane about z.
        let x1 = p[0] * self.cos_yaw - p[1] * self.sin_yaw;
        let y1 = p[0] * self.sin_yaw + p[1] * self.cos_yaw;
        let z1 = p[2];
        // Pitch: camera-frame right / up / depth for an eye above the plane.
        let x2 = x1;
        let y2 = y1 * self.cos_pitch + z1 * self.sin_pitch;
        let z2 = y1 * self.sin_pitch - z1 * self.cos_pitch;
        let z3 = z2 + self.distance;
        if z3 < NEAR_PLANE {
            return [f32::NAN, f32::NAN, f32::NAN];
        }
        let sx = FOCAL * x2 / z3;
        let sy = FOCAL * y2 / z3;
        let px = (sx + 1.0) * 0.5 * self.width;
        let py = (1.0 - (sy + 1.0) * 0.5) * self.height;
        [px, py, z3]
    }
}

/// Optional embellishments layered on top of the rasterized surface.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceRenderOpts {
    /// Draw mesh edges instead of filled triangles.
    pub show_wireframe: bool,
    /// Draw colored world axes from the origin (red=x, green=y, blue=z).
    /// They are depth-tested in the `z = 0` plane, so with the eye above the
    /// plane a filled surface hides all of the x and y axes but their ends
    /// past the mesh edge, and the z axis below the surface at the origin;
    /// the wireframe view shows them through the mesh.
    pub show_axes: bool,
    /// Draw a horizontal physical-scale reference in the lower-right.
    pub show_scale_bar: bool,
    /// Physical extent (Å) used to label the scale bar. Ignored when
    /// `show_scale_bar` is false.
    pub physical_extent: f32,
}

impl Default for SurfaceRenderOpts {
    fn default() -> Self {
        Self {
            show_wireframe: false,
            show_axes: false,
            show_scale_bar: false,
            physical_extent: 0.0,
        }
    }
}

/// Render a 2D heightmap as a 3D perspective-projected surface. Larger values
/// draw higher (see the module docs for the orientation convention).
pub fn render_surface_3d(
    data: &[f64],
    n: usize,
    width: usize,
    height: usize,
    camera: &Camera3D,
    colormap: fn(f64) -> [u8; 4],
) -> ColorImage {
    render_surface_3d_opts(
        data,
        n,
        width,
        height,
        camera,
        colormap,
        Color32::from_rgb(30, 30, 35),
        &SurfaceRenderOpts::default(),
    )
}

/// Render the surface with a caller-supplied background color. Matches the
/// pre-overlay API used across the app; forwards to `render_surface_3d_opts`.
pub fn render_surface_3d_with_bg(
    data: &[f64],
    n: usize,
    width: usize,
    height: usize,
    camera: &Camera3D,
    colormap: fn(f64) -> [u8; 4],
    bg_color: Color32,
) -> ColorImage {
    render_surface_3d_opts(
        data,
        n,
        width,
        height,
        camera,
        colormap,
        bg_color,
        &SurfaceRenderOpts::default(),
    )
}

/// Render the surface with full control over wireframe / axes / scale bar.
#[allow(clippy::too_many_arguments)]
pub fn render_surface_3d_opts(
    data: &[f64],
    n: usize,
    width: usize,
    height: usize,
    camera: &Camera3D,
    colormap: fn(f64) -> [u8; 4],
    bg_color: Color32,
    opts: &SurfaceRenderOpts,
) -> ColorImage {
    render_mesh(data, data, n, width, height, camera, colormap, bg_color, opts)
}

/// Render a 3D surface whose geometry (`heights`) and colormap value
/// (`colors`) come from different fields — e.g. curvature height colored by
/// gap magnitude. Both arrays are row-major n*n, pre-normalized to [0, 1].
#[allow(clippy::too_many_arguments)]
pub fn render_surface_3d_colored(
    heights: &[f64],
    colors: &[f64],
    n: usize,
    width: usize,
    height: usize,
    camera: &Camera3D,
    colormap: fn(f64) -> [u8; 4],
    bg_color: Color32,
    opts: &SurfaceRenderOpts,
) -> ColorImage {
    debug_assert_eq!(heights.len(), colors.len());
    render_mesh(heights, colors, n, width, height, camera, colormap, bg_color, opts)
}

/// Shared rasterizer core: `heights` drives vertex z displacement (upward, by
/// `HEIGHT_SCALE` for a value of 1.0), `colors` drives the colormap lookup;
/// both are row-major n*n arrays in [0, 1]. Row 0 sits at `y = -1` and column
/// 0 at `x = -1`.
#[allow(clippy::too_many_arguments)]
fn render_mesh(
    heights: &[f64],
    colors: &[f64],
    n: usize,
    width: usize,
    height: usize,
    camera: &Camera3D,
    colormap: fn(f64) -> [u8; 4],
    bg_color: Color32,
    opts: &SurfaceRenderOpts,
) -> ColorImage {
    let mut pixels = vec![bg_color; width * height];
    let mut zbuf = vec![f32::MAX; width * height];

    let step = (n / 64).max(1);
    let mesh_n = n.div_ceil(step);

    let mut verts: Vec<[f32; 3]> = Vec::with_capacity(mesh_n * mesh_n);
    let mut values: Vec<f64> = Vec::with_capacity(mesh_n * mesh_n);

    for iy in 0..mesh_n {
        let sy = (iy * step).min(n - 1);
        let y = (sy as f32 / n as f32) * 2.0 - 1.0;
        for ix in 0..mesh_n {
            let sx = (ix * step).min(n - 1);
            let x = (sx as f32 / n as f32) * 2.0 - 1.0;
            let idx = sy * n + sx;
            let z = heights[idx] as f32 * HEIGHT_SCALE;
            verts.push([x, y, z]);
            values.push(colors[idx]);
        }
    }

    let projector = Projector::new(camera, width, height);
    let transform = |p: [f32; 3]| -> [f32; 3] { projector.project(p) };

    let screen_verts: Vec<[f32; 3]> = verts.iter().map(|v| transform(*v)).collect();
    let light = normalize([0.3, 0.5, 0.8]);
    let wire_color = Color32::from_gray(220);

    for iy in 0..mesh_n - 1 {
        for ix in 0..mesh_n - 1 {
            let i00 = iy * mesh_n + ix;
            let i10 = iy * mesh_n + ix + 1;
            let i01 = (iy + 1) * mesh_n + ix;
            let i11 = (iy + 1) * mesh_n + ix + 1;

            let v00 = screen_verts[i00];
            let v10 = screen_verts[i10];
            let v01 = screen_verts[i01];
            let v11 = screen_verts[i11];

            if opts.show_wireframe {
                if !v00[0].is_nan() && !v10[0].is_nan() {
                    draw_line(&mut pixels, &mut zbuf, width, height, v00, v10, wire_color);
                }
                if !v00[0].is_nan() && !v01[0].is_nan() {
                    draw_line(&mut pixels, &mut zbuf, width, height, v00, v01, wire_color);
                }
                if ix == mesh_n - 2 && !v10[0].is_nan() && !v11[0].is_nan() {
                    draw_line(&mut pixels, &mut zbuf, width, height, v10, v11, wire_color);
                }
                if iy == mesh_n - 2 && !v01[0].is_nan() && !v11[0].is_nan() {
                    draw_line(&mut pixels, &mut zbuf, width, height, v01, v11, wire_color);
                }
            } else {
                let avg_val = (values[i00] + values[i10] + values[i01] + values[i11]) * 0.25;
                let [r, g, b, _] = colormap(avg_val);

                if !v00[0].is_nan() && !v10[0].is_nan() && !v01[0].is_nan() {
                    let normal = face_normal(verts[i00], verts[i10], verts[i01]);
                    let shade = shade_factor(normal, light);
                    let c = shade_color(r, g, b, shade);
                    rasterize_triangle(v00, v10, v01, c, &mut pixels, &mut zbuf, width, height);
                }
                if !v10[0].is_nan() && !v11[0].is_nan() && !v01[0].is_nan() {
                    let normal = face_normal(verts[i10], verts[i11], verts[i01]);
                    let shade = shade_factor(normal, light);
                    let c = shade_color(r, g, b, shade);
                    rasterize_triangle(v10, v11, v01, c, &mut pixels, &mut zbuf, width, height);
                }
            }
        }
    }

    if opts.show_axes {
        let origin = transform([0.0, 0.0, 0.0]);
        let ex = transform([AXIS_LEN, 0.0, 0.0]);
        let ey = transform([0.0, AXIS_LEN, 0.0]);
        let ez = transform([0.0, 0.0, AXIS_LEN * 0.6]);

        if !origin[0].is_nan() {
            if !ex[0].is_nan() {
                draw_line(
                    &mut pixels,
                    &mut zbuf,
                    width,
                    height,
                    origin,
                    ex,
                    Color32::from_rgb(240, 90, 90),
                );
            }
            if !ey[0].is_nan() {
                draw_line(
                    &mut pixels,
                    &mut zbuf,
                    width,
                    height,
                    origin,
                    ey,
                    Color32::from_rgb(90, 210, 120),
                );
            }
            if !ez[0].is_nan() {
                draw_line(
                    &mut pixels,
                    &mut zbuf,
                    width,
                    height,
                    origin,
                    ez,
                    Color32::from_rgb(120, 160, 255),
                );
            }
        }
    }

    if opts.show_scale_bar && opts.physical_extent > 0.0 {
        draw_scale_bar(&mut pixels, width, height, opts.physical_extent);
    }

    ColorImage {
        size: [width, height],
        pixels,
    }
}

/// Bresenham-style line with z-buffered pixels.
fn draw_line(
    pixels: &mut [Color32],
    zbuf: &mut [f32],
    width: usize,
    height: usize,
    a: [f32; 3],
    b: [f32; 3],
    color: Color32,
) {
    let dx = (b[0] - a[0]).abs();
    let dy = (b[1] - a[1]).abs();
    let steps = dx.max(dy).ceil() as i32;
    if steps <= 0 {
        return;
    }
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let px = a[0] + (b[0] - a[0]) * t;
        let py = a[1] + (b[1] - a[1]) * t;
        let pz = a[2] + (b[2] - a[2]) * t;
        if px < 0.0 || py < 0.0 {
            continue;
        }
        let ix = px as usize;
        let iy = py as usize;
        if ix >= width || iy >= height {
            continue;
        }
        // Slight bias so lines win ties with coplanar surface fills.
        let idx = iy * width + ix;
        if pz - 0.002 < zbuf[idx] {
            zbuf[idx] = pz - 0.002;
            pixels[idx] = color;
        }
    }
}

/// Draw a screen-space scale bar in the lower-right corner. The bar is 20 % of
/// the image width and labelled with the fraction of the physical extent it
/// spans (half the normalized data axis = physical_extent / 2).
fn draw_scale_bar(pixels: &mut [Color32], width: usize, height: usize, physical_extent: f32) {
    let bar_len = (width as f32 * 0.2) as i32;
    let margin = 16_i32;
    let y = height as i32 - margin;
    let x0 = width as i32 - margin - bar_len;
    let x1 = x0 + bar_len;
    let color = Color32::from_gray(230);

    if y < 0 || y >= height as i32 || x0 < 0 {
        return;
    }

    // Horizontal bar (2 px thick) + end ticks.
    for dy in -1..=0_i32 {
        let row = (y + dy) as usize;
        if row >= height {
            continue;
        }
        for xi in x0..=x1 {
            if xi < 0 || xi >= width as i32 {
                continue;
            }
            pixels[row * width + xi as usize] = color;
        }
    }
    for dy_tick in -4..=4_i32 {
        for &xt in &[x0, x1] {
            let r = y + dy_tick;
            if r < 0 || r >= height as i32 {
                continue;
            }
            if xt < 0 || xt >= width as i32 {
                continue;
            }
            pixels[r as usize * width + xt as usize] = color;
        }
    }

    // The normalized mesh goes from -1 to +1 in each axis, i.e. covers the
    // full physical_extent. The bar is 20 % of image width; pretend that
    // proportion maps to 20 % of the physical extent as a rough reference.
    let _angstroms = physical_extent * 0.2;
    // Labels would require a font renderer; skip and rely on the colorbar
    // axis ticks in the 2D view to communicate units. The tick marks still
    // give a visible reference length.
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

#[allow(clippy::too_many_arguments)]
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
    let min_x = v0[0].min(v1[0]).min(v2[0]).max(0.0) as i32;
    let max_x = v0[0].max(v1[0]).max(v2[0]).min(width as f32 - 1.0) as i32;
    let min_y = v0[1].min(v1[1]).min(v2[1]).max(0.0) as i32;
    let max_y = v0[1].max(v1[1]).max(v2[1]).min(height as f32 - 1.0) as i32;

    if min_x > max_x || min_y > max_y {
        return;
    }

    let area = edge_fn(v0, v1, v2);
    if area.abs() < 0.001 {
        return;
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
        assert!(
            non_bg > 100,
            "Expected rendered pixels, got only {} non-background",
            non_bg
        );
    }

    #[test]
    fn test_wireframe_leaves_mesh_interior_as_background() {
        let data = vec![0.5_f64; 32 * 32];
        let bg = Color32::from_rgb(30, 30, 35);
        let opts = SurfaceRenderOpts {
            show_wireframe: true,
            ..Default::default()
        };
        let img = render_surface_3d_opts(
            &data,
            32,
            128,
            128,
            &Camera3D::default(),
            |_| [200, 200, 200, 255],
            bg,
            &opts,
        );
        let non_bg = img.pixels.iter().filter(|&&p| p != bg).count();
        // Wireframe paints only edges — far fewer pixels than a filled surface.
        assert!(non_bg > 20 && non_bg < 128 * 128 / 2);
    }

    #[test]
    fn test_axes_adds_colored_pixels() {
        let data = vec![0.5_f64; 16 * 16];
        let bg = Color32::from_rgb(30, 30, 35);
        let opts = SurfaceRenderOpts {
            show_axes: true,
            ..Default::default()
        };
        let img = render_surface_3d_opts(
            &data,
            16,
            128,
            128,
            &Camera3D::default(),
            |_| [100, 100, 100, 255],
            bg,
            &opts,
        );
        // Red-channel dominant pixels should exist from the x-axis line.
        let red = img
            .pixels
            .iter()
            .filter(|p| {
                let [r, g, b, _] = p.to_array();
                r > 200 && g < 150 && b < 150
            })
            .count();
        assert!(red > 0, "Expected red x-axis pixels");
    }

    #[test]
    fn test_colored_render_produces_correct_size() {
        let heights = vec![0.5_f64; 32 * 32];
        let mut colors = vec![0.0_f64; 32 * 32];
        for i in 0..32 {
            for j in 0..32 {
                colors[i * 32 + j] = (i as f64 + j as f64) / 62.0;
            }
        }
        let bg = Color32::from_rgb(30, 30, 35);
        let img = render_surface_3d_colored(
            &heights, &colors, 32, 128, 128, &Camera3D::default(),
            |t| {
                let v = (t * 255.0) as u8;
                [v, v, v, 255]
            },
            bg, &SurfaceRenderOpts::default(),
        );
        assert_eq!(img.size, [128, 128]);
        assert_eq!(img.pixels.len(), 128 * 128);
        let non_bg = img.pixels.iter().filter(|&&p| p != bg).count();
        assert!(non_bg > 100, "Expected rendered pixels, got only {} non-background", non_bg);
    }

    #[test]
    fn test_colored_render_varies_with_color_array() {
        // Constant heights give a uniform face normal, so shading is constant;
        // distinct pixel colors can only come from the colors gradient.
        let heights = vec![0.5_f64; 32 * 32];
        let mut colors = vec![0.0_f64; 32 * 32];
        for i in 0..32 {
            for j in 0..32 {
                colors[i * 32 + j] = (i as f64 + j as f64) / 62.0;
            }
        }
        let bg = Color32::from_rgb(30, 30, 35);
        let img = render_surface_3d_colored(
            &heights, &colors, 32, 128, 128, &Camera3D::default(),
            |t| {
                let v = (t * 255.0) as u8;
                [v, v, v, 255]
            },
            bg, &SurfaceRenderOpts::default(),
        );
        let distinct: std::collections::HashSet<_> = img
            .pixels
            .iter()
            .filter(|&&p| p != bg)
            .map(|p| p.to_array())
            .collect();
        assert!(
            distinct.len() > 3,
            "Expected multiple distinct surface colors, got {}",
            distinct.len()
        );
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
        assert!((0.0..=1.0).contains(&s));
    }

    /// The capture bin's "tilt" camera (yaw 1.1, pitch 0.35, distance 3).
    fn tilt_camera() -> Camera3D {
        Camera3D {
            yaw: 1.1,
            pitch: 0.35,
            distance: 3.0,
        }
    }

    #[test]
    fn taller_vertex_projects_higher() {
        const SIZE: usize = 1024;
        let points = [
            (-1.0_f32, -1.0_f32),
            (1.0, -1.0),
            (-1.0, 1.0),
            (1.0, 1.0),
            (0.0, 0.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.5, -0.4),
            (-0.6, 0.3),
        ];
        for camera in [Camera3D::default(), tilt_camera()] {
            let projector = Projector::new(&camera, SIZE, SIZE);
            // In the yawed frame the eye sits at y1 = -distance·sin(pitch).
            // A point farther from the eye than that rises on screen as it
            // gets taller; a point nearer than the eye (the (-1, -1) corner,
            // almost under it and at or past the bottom edge of the frame)
            // sinks toward the eye's nadir instead. Points within 0.05 of
            // the boundary are not asserted either way.
            let eye_y1 = -camera.distance * camera.pitch.sin();
            let mut rising = 0;
            for (x, y) in points {
                let y1 = x * camera.yaw.sin() + y * camera.yaw.cos();
                let flat = projector.project([x, y, 0.0]);
                for h in [0.05_f32, HEIGHT_SCALE, 0.69] {
                    let tall = projector.project([x, y, h]);
                    assert!(
                        tall[2] < flat[2],
                        "({x}, {y}) at height {h} must be nearer than flat"
                    );
                    if y1 > eye_y1 + 0.05 {
                        assert!(
                            tall[1] < flat[1],
                            "({x}, {y}) at height {h} must draw higher: {} vs {}",
                            tall[1],
                            flat[1]
                        );
                        rising += 1;
                    } else if y1 < eye_y1 - 0.05 {
                        assert!(
                            tall[1] > flat[1],
                            "({x}, {y}) is nearer than the eye and must sink: {} vs {}",
                            tall[1],
                            flat[1]
                        );
                    }
                }
            }
            // Every point but the near corner rises, for both cameras.
            assert_eq!(rising, (points.len() - 1) * 3);
        }
    }

    #[test]
    fn flat_surface_framing_is_unchanged() {
        // Pixel positions of the flat plane's corners, origin and far edge for
        // the default camera in a 1024² frame, as the rasterizer drew them
        // before the height-sign fix (heights pointed downward). The fix flips
        // only z's contributions, so a flat surface must frame identically.
        let projector = Projector::new(&Camera3D::default(), 1024, 1024);
        let pinned = [
            ([0.0_f32, 0.0, 0.0], [512.0_f32, 512.0]),
            ([-1.0, -1.0, 0.0], [524.4268, 1214.3474]),
            ([1.0, -1.0, 0.0], [1088.5114, 505.05176]),
            ([-1.0, 1.0, 0.0], [-69.91406, 519.0134]),
            ([1.0, 1.0, 0.0], [505.58942, 149.6814]),
            ([0.0, 1.0, 0.0], [258.12042, 308.49573]),
        ];
        for (world, [px, py]) in pinned {
            let got = projector.project(world);
            assert!(
                (got[0] - px).abs() < 1e-3 && (got[1] - py).abs() < 1e-3,
                "{world:?} projected to ({}, {}), pinned ({px}, {py})",
                got[0],
                got[1]
            );
        }
    }

    #[test]
    fn z_axis_points_up() {
        const SIZE: usize = 128;
        let data = vec![0.0_f64; 16 * 16];
        let bg = Color32::from_rgb(30, 30, 35);
        let opts = SurfaceRenderOpts {
            show_axes: true,
            ..Default::default()
        };
        let img = render_surface_3d_opts(
            &data,
            16,
            SIZE,
            SIZE,
            &Camera3D::default(),
            |_| [100, 100, 100, 255],
            bg,
            &opts,
        );
        let blue = Color32::from_rgb(120, 160, 255);
        let (mut rows, mut count) = (0.0_f64, 0usize);
        for (i, p) in img.pixels.iter().enumerate() {
            if *p == blue {
                rows += (i / SIZE) as f64;
                count += 1;
            }
        }
        assert!(count > 5, "expected blue z-axis pixels, found {count}");
        let origin = Projector::new(&Camera3D::default(), SIZE, SIZE).project([0.0, 0.0, 0.0]);
        let centroid = rows / count as f64;
        assert!(
            centroid < f64::from(origin[1]) - 2.0,
            "z-axis centroid row {centroid} must be above the origin at row {}",
            origin[1]
        );
    }

    #[test]
    fn near_plane_is_clear_from_distance_1_5() {
        // Every reachable pitch (the viewport clamps to ±1.4), all yaws, the
        // four mesh corners at heights 0 and 1, and the three axis tips: no
        // vertex falls behind the near plane once distance ≥ 1.5. At the
        // 1.0 zoom limit the nearest corner can (see `Projector::project`).
        let tips = [
            [AXIS_LEN, 0.0, 0.0],
            [0.0, AXIS_LEN, 0.0],
            [0.0, 0.0, AXIS_LEN * 0.6],
        ];
        for distance in [1.5_f32, 2.0, 2.5, 5.0] {
            for i in 0..=280 {
                let pitch = -1.4 + i as f32 * 0.01;
                for j in 0..64 {
                    let yaw = j as f32 * std::f32::consts::TAU / 64.0;
                    let camera = Camera3D {
                        yaw,
                        pitch,
                        distance,
                    };
                    let projector = Projector::new(&camera, 256, 256);
                    for x in [-1.0_f32, 1.0] {
                        for y in [-1.0_f32, 1.0] {
                            for z in [0.0, HEIGHT_SCALE] {
                                assert!(
                                    projector.project([x, y, z])[2].is_finite(),
                                    "{camera:?} drops ({x}, {y}, {z})"
                                );
                            }
                        }
                    }
                    for tip in tips {
                        assert!(
                            projector.project(tip)[2].is_finite(),
                            "{camera:?} drops the axis tip {tip:?}"
                        );
                    }
                }
            }
        }
        let zoomed_in = Camera3D {
            yaw: 0.8,
            pitch: 0.6,
            distance: 1.0,
        };
        let projector = Projector::new(&zoomed_in, 256, 256);
        assert!(projector.project([-1.0, -1.0, 0.0])[2].is_finite());
        assert!(projector.project([-1.0, -1.0, HEIGHT_SCALE])[2].is_nan());
    }
}
