//! Headless Rust screenshot generator.
//!
//! Mirrors `scripts/capture_screenshots.py` for the Rust side: runs the same
//! physics the desktop app would, rasterizes the 3D surface via the software
//! renderer, layers the axes/overlay/scale bar, and writes PNGs into
//! `docs/images/` (or another directory via `--out-dir`).
//!
//! Unlike the desktop app this binary needs no display — it runs the CPU
//! rasterizer directly, so it works in CI and in headless sandboxes where
//! `cargo run -p moire-desktop` cannot open a window.
//!
//! Usage:
//!
//!     cargo run -p moire-desktop --bin capture -- [--out-dir docs/images]
//!                                                   [--only viewer-3d[,cooper-3d,...]]

use std::path::PathBuf;

use egui::Color32;

use moire_core::colormap::{coolwarm, viridis};
use moire_core::density::{compute_density_modulation, DensityConfig};
use moire_core::materials::LatticeType;
use moire_core::moire::{compute_moire, MoireConfig, MoireResult};
use moire_desktop::render::screenshot::save_color_image_to_png;
use moire_desktop::render::surface3d::{render_surface_3d_opts, Camera3D, SurfaceRenderOpts};

/// Paper-default preset: FeTe substrate + Sb2Te3 overlayer, θ=0°.
fn compute_moire_paper_default(grid: usize, extent: f64) -> MoireResult {
    let cfg = MoireConfig {
        substrate_a: 3.82,
        substrate_lattice_type: LatticeType::Square,
        overlayer_a: 4.264,
        overlayer_lattice_type: LatticeType::Hexagonal,
        twist_angle_deg: 0.0,
        resolution: grid,
        physical_extent: extent,
        dw_factor_substrate: 1.0,
        dw_factor_overlayer: 1.0,
    };
    compute_moire(&cfg).expect("moire pattern must compute for paper-default preset")
}

fn compute_gap_field(moire: &MoireResult) -> Vec<f64> {
    let cfg = DensityConfig::default();
    let result = compute_density_modulation(moire, &cfg).expect("density modulation must compute");
    result.gap_field
}

/// Normalise a slice to `[0, 1]`; required by the software rasterizer.
fn normalize(data: &[f64]) -> Option<Vec<f64>> {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &v in data {
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    let range = hi - lo;
    if range < 1.0e-15 {
        return None;
    }
    Some(data.iter().map(|&v| (v - lo) / range).collect())
}

/// Canonical camera angles matching the Python iso / 3D preset.
fn preset_camera(preset: &str) -> Camera3D {
    match preset {
        "iso" => Camera3D {
            yaw: 0.8,
            pitch: 0.6,
            distance: 2.5,
        },
        "tilt" => Camera3D {
            yaw: 1.1,
            pitch: 0.35,
            distance: 3.0,
        },
        "top" => Camera3D {
            yaw: 0.0,
            pitch: 1.25,
            distance: 2.8,
        },
        _ => Camera3D::default(),
    }
}

struct Scene {
    name: &'static str,
    filename: &'static str,
    build: fn() -> egui::ColorImage,
}

fn bg_dark() -> Color32 {
    Color32::from_rgb(18, 20, 28)
}

fn default_opts(extent: f64) -> SurfaceRenderOpts {
    SurfaceRenderOpts {
        show_wireframe: false,
        show_axes: true,
        show_scale_bar: true,
        physical_extent: extent as f32,
    }
}

fn scene_rust_desktop_2d() -> egui::ColorImage {
    let extent = 100.0_f64;
    let moire = compute_moire_paper_default(220, extent);
    let norm = normalize(&moire.pattern).expect("pattern must be normalizable");
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        1024,
        1024,
        &preset_camera("top"),
        viridis,
        bg_dark(),
        &default_opts(extent),
    )
}

fn scene_rust_desktop_3d() -> egui::ColorImage {
    let extent = 100.0_f64;
    let moire = compute_moire_paper_default(220, extent);
    let norm = normalize(&moire.pattern).expect("pattern must be normalizable");
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        1024,
        1024,
        &preset_camera("iso"),
        viridis,
        bg_dark(),
        &default_opts(extent),
    )
}

fn scene_rust_desktop_wireframe() -> egui::ColorImage {
    let extent = 100.0_f64;
    let moire = compute_moire_paper_default(220, extent);
    let norm = normalize(&moire.pattern).expect("pattern must be normalizable");
    let mut opts = default_opts(extent);
    opts.show_wireframe = true;
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        1024,
        1024,
        &preset_camera("iso"),
        viridis,
        bg_dark(),
        &opts,
    )
}

fn scene_rust_density_3d() -> egui::ColorImage {
    let extent = 100.0_f64;
    let moire = compute_moire_paper_default(220, extent);
    let density = compute_gap_field(&moire);
    let norm = normalize(&density).expect("density normalizable");
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        1024,
        1024,
        &preset_camera("tilt"),
        coolwarm,
        bg_dark(),
        &default_opts(extent),
    )
}

fn main() -> Result<(), String> {
    let mut out_dir = PathBuf::from("docs/images");
    let mut only: Vec<String> = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => {
                out_dir = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or_else(|| "--out-dir needs a path".to_string())?;
            }
            "--only" => {
                let raw = args
                    .next()
                    .ok_or_else(|| "--only needs a name".to_string())?;
                for name in raw.split(',') {
                    only.push(name.trim().to_string());
                }
            }
            "--help" | "-h" => {
                eprintln!(
                    "usage: cargo run -p moire-desktop --bin capture -- \
                     [--out-dir PATH] [--only scene[,scene...]]\n\
                     scenes: rust-desktop-2d, rust-desktop-3d, \
                     rust-desktop-wireframe, rust-density-3d"
                );
                return Ok(());
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let scenes: Vec<Scene> = vec![
        Scene {
            name: "rust-desktop-2d",
            filename: "rust-desktop-2d.png",
            build: scene_rust_desktop_2d,
        },
        Scene {
            name: "rust-desktop-3d",
            filename: "rust-desktop-3d.png",
            build: scene_rust_desktop_3d,
        },
        Scene {
            name: "rust-desktop-wireframe",
            filename: "rust-desktop-wireframe.png",
            build: scene_rust_desktop_wireframe,
        },
        Scene {
            name: "rust-density-3d",
            filename: "rust-density-3d.png",
            build: scene_rust_density_3d,
        },
    ];

    for scene in scenes {
        if !only.is_empty() && !only.iter().any(|n| n == scene.name) {
            continue;
        }
        eprintln!("[{}] rendering…", scene.name);
        let image = (scene.build)();
        let path = out_dir.join(scene.filename);
        save_color_image_to_png(&image, &path)?;
        eprintln!("[{}] wrote {}", scene.name, path.display());
    }

    Ok(())
}
