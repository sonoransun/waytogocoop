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
        crate::render::surface3d::render_surface_3d_opts(
            inputs.data,
            inputs.n,
            inputs.size[0],
            inputs.size[1],
            inputs.camera,
            inputs.colormap,
            inputs.background,
            inputs.opts,
        )
    }

    fn backend_name(&self) -> &'static str {
        "software-raster"
    }
}
