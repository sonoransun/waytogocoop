pub mod axes;
pub mod overlay;
pub mod pattern;
pub mod renderer3d;
pub mod screenshot;
pub mod surface3d;

/// wgpu / egui-wgpu 3D renderer — opt-in via the `gpu` feature.
#[cfg(feature = "gpu")]
pub mod gpu;

pub use renderer3d::{FrameInputs, Renderer3D, SoftwareRenderer};
