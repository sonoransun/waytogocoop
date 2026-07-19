//! Backend-agnostic contract for the 3D surface renderer.
//!
//! Two concrete backends satisfy this contract:
//!
//! - The CPU software rasterizer in [`crate::render::surface3d`] — default.
//! - A wgpu/egui-wgpu GPU renderer in [`crate::render::gpu`] — behind the
//!   `gpu` cargo feature.
//!
//! The GPU path is scaffolding for the rewrite described in `CONTRIBUTING.md`
//! ("wgpu 3D renderer"). Its shader, mesh, camera, and readback modules are
//! in place; runtime integration with `app::rerender_surface` is feature-
//! gated so CI, headless environments, and machines without a working Vulkan/
//! Metal/DX12 stack keep running on the software path.

use egui::{Color32, ColorImage};

/// Options bundle that captures user toggles layered on top of the mesh.
pub use crate::render::surface3d::{Camera3D, SurfaceRenderOpts};

/// Inputs required to produce one 3D surface frame.
///
/// Kept as a borrowed struct so both backends can process without allocating.
#[derive(Debug, Clone, Copy)]
pub struct FrameInputs<'a> {
    /// Row-major height values in `[0, 1]`, length `n*n`.
    pub data: &'a [f64],
    /// Grid resolution (NxN).
    pub n: usize,
    /// Output texture size in pixels.
    pub size: [usize; 2],
    /// Camera state.
    pub camera: &'a Camera3D,
    /// Colormap sampled per vertex / per pixel.
    pub colormap: fn(f64) -> [u8; 4],
    /// Background color outside the mesh silhouette.
    pub background: Color32,
    /// Overlay toggles.
    pub opts: &'a SurfaceRenderOpts,
    /// Optional world-space clip-plane in normalized height coords (`[-1, 1]`);
    /// vertices above this plane are discarded. `None` disables clipping.
    pub clip_z: Option<f32>,
    /// Optional per-vertex color field (row-major `n*n`, in `[0, 1]`) sampled
    /// through `colormap` while `data` drives the height. `None` colors by
    /// height — the flat-grid default; `Some` is the curved-sheet path where
    /// geometry and color come from different fields.
    pub color_data: Option<&'a [f64]>,
}

/// Minimal trait both backends satisfy.
///
/// The software backend is stateless — every call produces a fresh image.
/// The GPU backend owns persistent buffers (mesh VBO, depth target, LUT
/// texture) across calls so it can update individual pieces without
/// reuploading everything.
pub trait Renderer3D {
    /// Render a single frame of the inputs to a `ColorImage`.
    fn render(&mut self, inputs: FrameInputs<'_>) -> ColorImage;

    /// Short name for logs/telemetry.
    fn backend_name(&self) -> &'static str;
}

/// The software implementation of [`Renderer3D`] — a stateless adapter over
/// the existing [`crate::render::surface3d::render_surface_3d_opts`]. Kept in
/// this file so downstream code can refer to `renderer3d::SoftwareRenderer`
/// uniformly regardless of whether the `gpu` feature is enabled.
#[derive(Default)]
pub struct SoftwareRenderer;

impl Renderer3D for SoftwareRenderer {
    fn render(&mut self, inputs: FrameInputs<'_>) -> ColorImage {
        // The software path ignores `clip_z` for now; the GPU path consumes it
        // via a WGSL uniform. When the software renderer learns to clip it
        // should branch here.
        let _ = inputs.clip_z;
        match inputs.color_data {
            // Curved-sheet path: geometry from `data`, color from `color_data`.
            Some(colors) => crate::render::surface3d::render_surface_3d_colored(
                inputs.data,
                colors,
                inputs.n,
                inputs.size[0],
                inputs.size[1],
                inputs.camera,
                inputs.colormap,
                inputs.background,
                inputs.opts,
            ),
            None => crate::render::surface3d::render_surface_3d_opts(
                inputs.data,
                inputs.n,
                inputs.size[0],
                inputs.size[1],
                inputs.camera,
                inputs.colormap,
                inputs.background,
                inputs.opts,
            ),
        }
    }

    fn backend_name(&self) -> &'static str {
        "software-raster"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::surface3d::{render_surface_3d_colored, render_surface_3d_opts};

    fn ramp(n: usize) -> Vec<f64> {
        (0..n * n).map(|i| i as f64 / (n * n) as f64).collect()
    }

    #[test]
    fn test_software_renderer_matches_direct_call() {
        let n = 16;
        let data = ramp(n);
        let camera = Camera3D::default();
        let opts = SurfaceRenderOpts::default();
        let bg = Color32::from_rgb(10, 20, 30);

        let mut renderer = SoftwareRenderer;
        let via_trait = renderer.render(FrameInputs {
            data: &data,
            n,
            size: [64, 64],
            camera: &camera,
            colormap: moire_core::colormap::viridis,
            background: bg,
            opts: &opts,
            clip_z: None,
            color_data: None,
        });
        let direct = render_surface_3d_opts(
            &data,
            n,
            64,
            64,
            &camera,
            moire_core::colormap::viridis,
            bg,
            &opts,
        );

        assert_eq!(via_trait.size, direct.size);
        assert_eq!(via_trait.pixels, direct.pixels);
    }

    #[test]
    fn test_colored_dispatch() {
        let n = 16;
        let heights = ramp(n);
        // A distinct color field so a wrong dispatch would visibly differ.
        let colors: Vec<f64> = heights.iter().rev().copied().collect();
        let camera = Camera3D::default();
        let opts = SurfaceRenderOpts::default();
        let bg = Color32::from_rgb(10, 20, 30);

        let mut renderer = SoftwareRenderer;
        let via_trait = renderer.render(FrameInputs {
            data: &heights,
            n,
            size: [64, 64],
            camera: &camera,
            colormap: moire_core::colormap::coolwarm,
            background: bg,
            opts: &opts,
            clip_z: None,
            color_data: Some(&colors),
        });
        let direct = render_surface_3d_colored(
            &heights,
            &colors,
            n,
            64,
            64,
            &camera,
            moire_core::colormap::coolwarm,
            bg,
            &opts,
        );

        assert_eq!(via_trait.pixels, direct.pixels);
    }
}
