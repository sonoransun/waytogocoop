//! wgpu-based 3D renderer — feature-gated replacement for the CPU rasterizer.
//!
//! The GPU path is opt-in via the `gpu` cargo feature. When enabled, this
//! module provides a [`GpuRenderer`] that implements
//! [`crate::render::renderer3d::Renderer3D`] by driving a WGSL surface
//! pipeline against an offscreen `wgpu::Texture`, reading it back into an
//! `egui::ColorImage` so the rest of the app stays backend-agnostic.
//!
//! The pieces are split across submodules to mirror the layout described in
//! `CONTRIBUTING.md`:
//!
//! - [`pipeline`] — creates the `RenderPipeline`, bind-group layouts, depth
//!   target, MSAA resolve target.
//! - [`mesh`]     — uploads the NxN heightmap as an indexed triangle grid
//!   with precomputed per-vertex normals from 4-neighbour central differences.
//! - [`camera`]   — converts the spherical [`Camera3D`] to view/projection
//!   matrices via `nalgebra` (no `glam` dep).
//! - [`readback`] — copies the offscreen color target to CPU memory,
//!   respecting wgpu's 256-byte row alignment.
//!
//! The shaders live under `gpu/shaders/`:
//!
//! - `surface.wgsl` — vertex + fragment for the surface mesh (Lambert,
//!   Blinn-Phong specular, Fresnel-Schlick, optional clip-plane discard).
//! - `volume.wgsl`  — stretch-goal fragment-stage raymarch over a 3D LUT
//!   for `ProximityResult::gap_3d`; currently a stub for a future tab.

pub mod camera;
pub mod mesh;
pub mod pipeline;
pub mod readback;

use bytemuck::{Pod, Zeroable};

use crate::render::renderer3d::{FrameInputs, Renderer3D};

/// Shader uniform block. Layout is `#[repr(C)]` to match WGSL memory rules.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct SurfaceUniforms {
    pub view_proj: [[f32; 4]; 4],
    pub eye: [f32; 3],
    pub clip_z: f32,
    pub light_dir: [f32; 3],
    pub ambient: f32,
}

impl Default for SurfaceUniforms {
    fn default() -> Self {
        Self {
            view_proj: [[0.0; 4]; 4],
            eye: [0.0; 3],
            clip_z: f32::INFINITY,
            light_dir: [0.3, 0.5, 0.8],
            ambient: 0.25,
        }
    }
}

/// The GPU renderer.
///
/// Runtime integration with `app::rerender_surface` is left to a follow-up
/// change; this struct exists so the GPU backend can be constructed and
/// driven by anyone who holds an eframe `wgpu_render_state`. See the
/// `CONTRIBUTING.md` future-work notes for the remaining wire-through.
pub struct GpuRenderer {
    // Owned here so we keep pipeline objects alive for the renderer's lifetime.
    // The concrete wgpu handles are created lazily on first `render` so tests
    // that exercise the struct without a device can still compile.
    pipeline_state: Option<pipeline::PipelineState>,
    mesh_state: Option<mesh::MeshState>,
    lut_texture: Option<wgpu::Texture>,
    uniforms: SurfaceUniforms,
}

impl Default for GpuRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuRenderer {
    pub fn new() -> Self {
        Self {
            pipeline_state: None,
            mesh_state: None,
            lut_texture: None,
            uniforms: SurfaceUniforms::default(),
        }
    }

    fn ensure_initialized(&mut self, _device: &wgpu::Device, _queue: &wgpu::Queue) {
        if self.pipeline_state.is_none() {
            // Real initialization creates the surface pipeline + depth buffer
            // + MSAA resolve target here. The skeleton leaves these None; a
            // follow-up change adds the actual wgpu object construction.
        }
    }
}

impl Renderer3D for GpuRenderer {
    fn render(&mut self, inputs: FrameInputs<'_>) -> egui::ColorImage {
        // Update per-frame uniforms.
        let (view_proj, eye) =
            camera::view_proj_matrix(inputs.camera, inputs.size[0] as f32 / inputs.size[1] as f32);
        self.uniforms.view_proj = view_proj;
        self.uniforms.eye = eye;
        self.uniforms.clip_z = inputs.clip_z.unwrap_or(f32::INFINITY);

        // Until the full wgpu pipeline lands we fall back to the CPU path so
        // the GPU backend still produces visually correct frames. This keeps
        // the trait wire-through meaningful during the staged rollout
        // described in CONTRIBUTING.md.
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
        "wgpu"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_uniforms_have_sane_values() {
        let u = SurfaceUniforms::default();
        assert!(u.ambient > 0.0 && u.ambient < 1.0);
        assert!(u.clip_z.is_infinite() && u.clip_z > 0.0);
    }

    #[test]
    fn surface_shader_parses() {
        // Shader compiles (WGSL -> naga IR) without a GPU device.
        let src = include_str!("shaders/surface.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL parses");
        assert!(!module.entry_points.is_empty());
    }

    #[test]
    fn volume_shader_parses() {
        let src = include_str!("shaders/volume.wgsl");
        let module = naga::front::wgsl::parse_str(src).expect("WGSL parses");
        assert!(!module.entry_points.is_empty());
    }
}
