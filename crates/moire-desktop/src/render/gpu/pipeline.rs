//! wgpu render-pipeline construction for the surface shader.
//!
//! Owns bind-group layouts, the depth texture, and the 4x MSAA color target.
//! A follow-up change adds the actual pipeline creation inside
//! `PipelineState::new`; for now the type exists so [`GpuRenderer`] can hold
//! it once the wire-through lands.

/// GPU pipeline state.
pub struct PipelineState {
    pub pipeline: wgpu::RenderPipeline,
    pub depth_view: wgpu::TextureView,
    pub color_view: wgpu::TextureView, // resolve target (1x)
    pub msaa_view: wgpu::TextureView,  // multisample target (4x)
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub sample_count: u32,
}

/// Shader source shared with tests via `include_str!`.
pub const SURFACE_WGSL: &str = include_str!("shaders/surface.wgsl");
pub const VOLUME_WGSL: &str = include_str!("shaders/volume.wgsl");

/// Sample count for MSAA; 4x is the universally supported rate.
pub const MSAA_SAMPLE_COUNT: u32 = 4;

/// Depth + color format for offscreen targets. `Depth32Float` works on all
/// backends (including WebGPU) and avoids the `Depth24` family's Metal quirks.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
