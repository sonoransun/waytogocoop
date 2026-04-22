//! Library crate for `moire-desktop`.
//!
//! Exposes the `render`, `app`, and `ui` modules so additional binaries
//! (`src/bin/capture.rs`) can drive the software rasterizer headlessly for
//! reproducible screenshot generation. The main `moire-desktop` binary
//! (`main.rs`) re-exports these through the same paths to keep its imports
//! unchanged.

pub mod app;
pub mod render;
pub mod ui;
