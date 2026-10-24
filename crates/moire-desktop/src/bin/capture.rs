//! Headless generator for the desktop-app figures in `docs/images/`.
//!
//! The Rust twin of `scripts/capture_screenshots.py`. Each scene runs the same
//! `moire-core` physics as the desktop app and draws it with the app's own
//! code paths: the `render::surface3d` CPU rasterizer for 3D views, and a CPU
//! copy of the egui texture draw for 2D views. No window or GPU is needed, so
//! it runs in CI and headless sandboxes. Every PNG is 1024 x 1024 and opaque;
//! 3D views sit on the app's dark-mode background. The figure gallery
//! (`docs/images/README.md`) holds the canonical captions.
//!
//! Scene kinds use the same labels as the Python registry:
//! - `app`: what the desktop app draws at settings reachable in its UI.
//! - `presentation`: app physics plus exactly one named display step.
//!
//! Extents are FULL widths. A Rust `physical_extent` of 200 Å spans
//! -100..+100 Å, so it equals the web app's "±100 Å" window: the web extent is
//! a half-width. The desktop default is 256² samples over 200 Å.
//!
//! ```text
//! cargo run --release -p moire-desktop --bin capture              # every scene
//! cargo run --release -p moire-desktop --bin capture -- --list    # the registry
//! cargo run --release -p moire-desktop --bin capture -- --only rust-desktop-2d,rust-graphene-pattern
//! cargo run --release -p moire-desktop --bin capture -- --out-dir /tmp/figs
//! ```
//!
//! The default output directory is this repository's `docs/images`, resolved
//! from the crate manifest, so the working directory does not matter.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use egui::{Color32, ColorImage};

use moire_core::colormap::{coolwarm, viridis};
use moire_core::curvature::{compute_curvature, CurvatureConfig, CurvatureGeometry};
use moire_core::density::{compute_density_modulation, DensityConfig};
use moire_core::graphene::{
    compute_graphene_stack_v2, GrapheneStackConfig, GrapheneStackConfigV2, StackingKind, GRAPHENE_A,
};
use moire_core::materials::LatticeType;
use moire_core::moire::{compute_moire, MoireConfig, MoireResult};
use moire_core::pdw::{
    compute_pdw_state, moire_locked_pdw_ratio, moire_locked_period, pdw_real_space, PdwConfig,
    TC_FETE_FILM,
};
use moire_core::spin_fluctuation::{glue_modulated_gap, SfCoherence, SfGlueConfig};
use moire_desktop::render::surface3d::{
    render_surface_3d_colored, render_surface_3d_opts, Camera3D, SurfaceRenderOpts,
};

/// Default output directory: the repository's `docs/images`.
const DEFAULT_OUT_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/images");

/// Side of every capture in pixels (the docs' "rust" frame).
const SIZE: usize = 1024;

/// Frame label printed by `--list`, beside the Python frames "wide" and "square".
const FRAME: &str = "rust";

/// Docs figure budget: a soft cap per PNG and a cap for the whole folder.
const SOFT_CAP_BYTES: u64 = 800 * 1024;
const FOLDER_BUDGET_BYTES: u64 = 12 * 1024 * 1024;

/// Desktop default grid (`MoireApp::default`: 256² samples).
const DESKTOP_GRID: usize = 256;

/// Desktop default window (`MoireApp::default`: 200 Å full width = ±100 Å).
const DESKTOP_EXTENT: f64 = 200.0;

/// Window of the 3D moire scenes: 50 Å full width (±25 Å), the minimum of
/// the desktop extent slider. `render_mesh` keeps every (n/64)-th sample with
/// no prefilter, so 256² samples become a 64-vertex mesh. At 50 Å its
/// 0.78 Å vertex spacing (Nyquist 4.0 Å⁻¹) resolves the 3.7 Å rows and their
/// 1.9 Å sum terms (3.35 Å⁻¹); at the 200 Å default the 3.1 Å spacing
/// aliased them.
const MESH_WINDOW: f64 = 50.0;

/// One docs figure. `filename` stays a string literal: the docs linter
/// (`scripts/check_docs.py`) reads `filename: "….png"` from this file.
struct Scene {
    name: &'static str,
    filename: &'static str,
    /// `app` or `presentation` (see the module docs).
    kind: &'static str,
    build: fn() -> ColorImage,
}

/// The registry, in canonical docs order: moire, gap, graphene stacks,
/// curvature, spin-fluctuation pairing, pair density waves.
const SCENES: &[Scene] = &[
    Scene {
        name: "rust-desktop-2d",
        filename: "rust-desktop-2d.png",
        kind: "app",
        build: scene_rust_desktop_2d,
    },
    Scene {
        name: "rust-desktop-3d",
        filename: "rust-desktop-3d.png",
        kind: "app",
        build: scene_rust_desktop_3d,
    },
    Scene {
        name: "rust-desktop-wireframe",
        filename: "rust-desktop-wireframe.png",
        kind: "app",
        build: scene_rust_desktop_wireframe,
    },
    Scene {
        name: "rust-density-3d",
        filename: "rust-density-3d.png",
        kind: "app",
        build: scene_rust_density_3d,
    },
    Scene {
        name: "rust-graphene-pattern",
        filename: "rust-graphene-pattern.png",
        kind: "presentation",
        build: scene_rust_graphene_pattern,
    },
    Scene {
        name: "rust-curved-3d",
        filename: "rust-curved-3d.png",
        kind: "app",
        build: scene_rust_curved_3d,
    },
    Scene {
        name: "rust-sf-glue-3d",
        filename: "rust-sf-glue-3d.png",
        kind: "app",
        build: scene_rust_sf_glue_3d,
    },
    Scene {
        name: "rust-pdw-whatif-3d",
        filename: "rust-pdw-whatif-3d.png",
        kind: "app",
        build: scene_rust_pdw_whatif_3d,
    },
];

// ---------------------------------------------------------------------------
// Shared physics and drawing helpers
// ---------------------------------------------------------------------------

/// Desktop defaults: FeTe substrate (square, 3.82 Å) under Sb₂Te₃
/// (hexagonal, 4.264 Å), untwisted, no isotope substitution.
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

/// Normalise a slice to `[0, 1]`, as the app does before colormapping.
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

/// Background of the desktop app's dark-mode 3D view (`surface_bg` in
/// `app.rs`), so captures look like the running app.
fn bg_dark() -> Color32 {
    Color32::from_rgb(30, 30, 35)
}

/// The desktop app's default 3D camera (`Camera3D::default()`: yaw 0.8,
/// pitch 0.6, distance 2.5). Pitch is the tilt away from a plan view: 0 looks
/// along the height axis and larger values approach the horizon. (The old
/// "top" preset, pitch 1.25 rad, was a grazing 72° view and is gone.)
///
/// As in the app, `render_mesh` projects larger heights nearer the camera
/// and higher on screen (its blue +z axis points up), so maxima of the
/// plotted field stand up in every 3D capture. The red and green x, y axes
/// lie in the z = 0 plane under the surface, so a filled view shows little
/// more than their stubs past the mesh edge. At this distance the mesh's
/// left, right and front corners run past the frame.
fn iso_camera() -> Camera3D {
    Camera3D::default()
}

/// A steeper, slightly more distant view than `iso_camera` (yaw 1.1, pitch
/// 0.35 rad = 20° from a plan view, distance 3.0); the app reaches it by
/// dragging and scrolling.
fn tilt_camera() -> Camera3D {
    Camera3D {
        yaw: 1.1,
        pitch: 0.35,
        distance: 3.0,
    }
}

/// The app's default 3D overlays: world axes and the unlabelled scale bar.
fn default_opts(extent: f64) -> SurfaceRenderOpts {
    SurfaceRenderOpts {
        show_wireframe: false,
        show_axes: true,
        show_scale_bar: true,
        physical_extent: extent as f32,
    }
}

/// sRGB-encoded 8-bit channel to linear light.
fn linear_from_srgb(c: u8) -> f64 {
    let c = f64::from(c) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light to an sRGB-encoded 8-bit channel, rounded to nearest.
fn srgb_from_linear(l: f64) -> u8 {
    let c = if l < 0.0031308 {
        12.92 * l
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    };
    (c * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Draw an `n x n` field into a `size x size` image the way the desktop 2D
/// view does, without the axes and colorbar that egui paints live
/// (`render/axes.rs`), which a headless capture cannot include.
///
/// - Orientation: `render/pattern.rs` uploads data row 0 (y = -extent/2) as
///   the texture's top row, and `ui/viewport.rs` paints it with v flipped
///   (uv (0,1)-(1,0)), so row 0 (y_min) is at the bottom and column 0 on
///   the left, as on the web heatmaps and under the 2D axes' labels.
/// - Filtering: `TextureOptions::LINEAR` with clamp-to-edge. egui keeps
///   textures in sRGB, so the GPU blends texels in linear light and the
///   shader re-encodes to sRGB. That is reproduced here, without egui's
///   sub-level dithering. When `n == size` every pixel is one texel.
fn render_texture_2d(
    data: &[f64],
    n: usize,
    size: usize,
    colormap: fn(f64) -> [u8; 4],
) -> ColorImage {
    let texels: Vec<[f64; 3]> = data
        .iter()
        .map(|&v| {
            let [r, g, b, _] = colormap(v);
            [
                linear_from_srgb(r),
                linear_from_srgb(g),
                linear_from_srgb(b),
            ]
        })
        .collect();
    // Bilinear taps for each output row/column; texel centres sit at
    // (i + 0.5) / n of the image.
    let last = (n - 1) as f64;
    let taps: Vec<(usize, usize, f64)> = (0..size)
        .map(|p| {
            let t = ((p as f64 + 0.5) * n as f64 / size as f64 - 0.5).clamp(0.0, last);
            let i0 = t.floor() as usize;
            (i0, (i0 + 1).min(n - 1), t - i0 as f64)
        })
        .collect();
    let mut pixels = Vec::with_capacity(size * size);
    // Output rows run from the top of the image; data row 0 goes to the bottom.
    for &(y0, y1, fy) in taps.iter().rev() {
        for &(x0, x1, fx) in &taps {
            let mut rgb = [0_u8; 3];
            for (c, out) in rgb.iter_mut().enumerate() {
                let top = texels[y0 * n + x0][c] * (1.0 - fx) + texels[y0 * n + x1][c] * fx;
                let bottom = texels[y1 * n + x0][c] * (1.0 - fx) + texels[y1 * n + x1][c] * fx;
                *out = srgb_from_linear(top * (1.0 - fy) + bottom * fy);
            }
            pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
        }
    }
    ColorImage {
        size: [size, size],
        pixels,
    }
}

/// Radius in samples of a Gaussian kernel truncated at 3σ.
fn blur_radius(sigma: f64) -> usize {
    (3.0 * sigma).ceil() as usize
}

/// Separable Gaussian blur of an `n x n` grid (`sigma` in samples, kernel
/// truncated at 3σ). Returns only the `(n - 2r)²` interior, where the kernel
/// stays inside the grid, so no edge rule is needed: pad the input by
/// `r = blur_radius(sigma)` samples on every side.
fn gaussian_blur_interior(data: &[f64], n: usize, sigma: f64) -> Vec<f64> {
    let r = blur_radius(sigma);
    let m = n - 2 * r;
    let raw: Vec<f64> = (0..=2 * r)
        .map(|k| {
            let d = k as f64 - r as f64;
            (-d * d / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();

    // Horizontal pass over every row, keeping the interior columns.
    let mut rows = vec![0.0; n * m];
    for (src, dst) in data.chunks_exact(n).zip(rows.chunks_exact_mut(m)) {
        for (d, window) in dst.iter_mut().zip(src.windows(2 * r + 1)) {
            *d = weights.iter().zip(window).map(|(w, v)| w * v).sum();
        }
    }
    // Vertical pass, keeping the interior rows.
    let mut out = vec![0.0; m * m];
    for (y, dst) in out.chunks_exact_mut(m).enumerate() {
        for (k, w) in weights.iter().enumerate() {
            let src = &rows[(y + k) * m..(y + k + 1) * m];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += w * s;
            }
        }
    }
    out
}

/// Bilinear resample of an `n x n` grid onto `m x m`, corner-aligned: both
/// grids span the same window, like the physics grids (sample i sits at
/// i / (n - 1) of the extent).
fn resample_bilinear(data: &[f64], n: usize, m: usize) -> Vec<f64> {
    let scale = (n - 1) as f64 / (m - 1) as f64;
    let taps: Vec<(usize, f64)> = (0..m)
        .map(|j| {
            let t = j as f64 * scale;
            let i0 = (t.floor() as usize).min(n - 2);
            (i0, t - i0 as f64)
        })
        .collect();
    let mut out = Vec::with_capacity(m * m);
    for &(y0, fy) in &taps {
        for &(x0, fx) in &taps {
            let at = |y: usize, x: usize| data[y * n + x];
            let top = at(y0, x0) * (1.0 - fx) + at(y0, x0 + 1) * fx;
            let bottom = at(y0 + 1, x0) * (1.0 - fx) + at(y0 + 1, x0 + 1) * fx;
            out.push(top * (1.0 - fy) + bottom * fy);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

/// App view: the Moire Pattern tab in 2D at the desktop defaults (FeTe /
/// Sb₂Te₃, θ = 0°, 256² samples over 200 Å = ±100 Å, viridis), magnified 4×
/// with the app's bilinear texture filtering.
fn scene_rust_desktop_2d() -> ColorImage {
    let moire = compute_moire_paper_default(DESKTOP_GRID, DESKTOP_EXTENT);
    render_texture_2d(&moire.pattern, moire.resolution, SIZE, viridis)
}

/// App view: the Moire Pattern tab in 3D (viridis, default camera) in the
/// 50 Å window.
fn scene_rust_desktop_3d() -> ColorImage {
    let moire = compute_moire_paper_default(DESKTOP_GRID, MESH_WINDOW);
    let norm = normalize(&moire.pattern).expect("pattern must be normalizable");
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        SIZE,
        SIZE,
        &iso_camera(),
        viridis,
        bg_dark(),
        &default_opts(MESH_WINDOW),
    )
}

/// App view: as `rust-desktop-3d` with the Wireframe toggle on (mesh edges
/// only, in the app's fixed light grey; no colormap).
fn scene_rust_desktop_wireframe() -> ColorImage {
    let moire = compute_moire_paper_default(DESKTOP_GRID, MESH_WINDOW);
    let norm = normalize(&moire.pattern).expect("pattern must be normalizable");
    let mut opts = default_opts(MESH_WINDOW);
    opts.show_wireframe = true;
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        SIZE,
        SIZE,
        &iso_camera(),
        viridis,
        bg_dark(),
        &opts,
    )
}

/// App view: the Density Modulation tab in 3D, the gap field Δ(r) at the
/// desktop default amplitude (15 % of Δ_avg, so 2.63-3.55 meV), normalised,
/// coolwarm (high Δ red), default camera, 50 Å window.
fn scene_rust_density_3d() -> ColorImage {
    let moire = compute_moire_paper_default(DESKTOP_GRID, MESH_WINDOW);
    let density = compute_gap_field(&moire);
    let norm = normalize(&density).expect("density normalizable");
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        SIZE,
        SIZE,
        &iso_camera(),
        coolwarm,
        bg_dark(),
        &default_opts(MESH_WINDOW),
    )
}

/// Presentation: magic-angle twisted bilayer graphene (1.08°) over 400 Å
/// (±200 Å, the web graphene window) as a 2D texture (viridis, row 0 at the
/// bottom like the app). The 2.13 Å atomic rows need sampling finer than
/// 0.53 Å, so the stack is computed at 0.25 Å: 1601² samples over 400 Å, plus a
/// margin of one blur radius (4.5 Å) so the blur never reaches past the
/// window. The named display step is a Gaussian average with σ = 1.5 Å,
/// which removes the atomic rows and keeps the 130.5 Å moire envelope. The
/// result is resampled to the 1024² texture and normalised.
fn scene_rust_graphene_pattern() -> ColorImage {
    const N: usize = 1601;
    const EXTENT: f64 = 400.0;
    const SIGMA_ANGSTROM: f64 = 1.5;
    let dx = EXTENT / (N - 1) as f64;
    let sigma = SIGMA_ANGSTROM / dx;
    let pad = blur_radius(sigma);
    let n_padded = N + 2 * pad;
    let cfg = GrapheneStackConfigV2 {
        base: GrapheneStackConfig {
            stacking: StackingKind::TwistedBilayer,
            twist_angle_deg: 1.08,
            lattice_a: GRAPHENE_A,
            resolution: n_padded,
            physical_extent: dx * (n_padded - 1) as f64,
        },
        ..GrapheneStackConfigV2::default()
    };
    let stack = compute_graphene_stack_v2(&cfg, None)
        .expect("graphene stack must compute for magic-angle preset");
    let envelope = gaussian_blur_interior(&stack.pattern, n_padded, sigma);
    let display =
        normalize(&resample_bilinear(&envelope, N, SIZE)).expect("moire envelope must vary");
    render_texture_2d(&display, SIZE, SIZE, viridis)
}

/// Curved-sheet scene: Gaussian-bump height field (5 Å, σ = 50 Å, 220² over
/// 200 Å) colored by the signed pseudo-magnetic field B_ps of the K valley
/// (coolwarm per the signed-quantity convention; the field is symmetric,
/// ±5.7 T, so zero sits mid-scale). Exercises the colored-surface render
/// path (geometry ≠ colormap field). Tilt camera.
fn scene_rust_curved_3d() -> ColorImage {
    let extent = 200.0_f64;
    let cfg = CurvatureConfig {
        geometry: CurvatureGeometry::GaussianBump,
        amplitude: 5.0,
        sigma: 50.0,
        resolution: 220,
        physical_extent: extent,
        ..CurvatureConfig::default()
    };
    let curved = compute_curvature(&cfg).expect("curvature must compute for Gaussian bump");
    let heights = normalize(&curved.height).expect("height field must be normalizable");
    let colors = normalize(&curved.pseudo_field).expect("pseudo-field must be normalizable");
    render_surface_3d_colored(
        &heights,
        &colors,
        curved.resolution,
        SIZE,
        SIZE,
        &tilt_camera(),
        coolwarm,
        bg_dark(),
        &default_opts(extent),
    )
}

/// Moire glue what-if (SPECULATIVE): the Spin Fluctuations tab's Glue gap
/// view in 3D at the web `sf-glue-imprint` scene's operating point. The
/// coherence-dressed gap Δ(r) for Sb₂Te₃ on FeTe (η = 0.1, Tc target 5 K,
/// clean coherence, the default power-law pairing map, L_m = the shared
/// two-lattice period) on the desktop's 256² pattern over the 50 Å window,
/// normalised, coolwarm (red = larger Δ), tilt camera. The dressed Δ spans
/// about 0.88-1.23 meV; its maxima stand up as red peaks. The dressing only
/// scales Δ_loc − Δ̄ by F, so once normalised the image is the local Tc(r)
/// map, identical for every coherence model. Its blue flecks are the
/// sub-ξ dips of that local estimate (about 1 Å wide; ξ = 54 Å), the deepest
/// to Tc(r) = 0: local-limit features, which the gallery captions flag.
/// Drawn to scale they sink below the surrounding relief, so most of them
/// show only as blue facets between the peaks.
fn scene_rust_sf_glue_3d() -> ColorImage {
    let moire = compute_moire_paper_default(DESKTOP_GRID, MESH_WINDOW);
    let period = moire_locked_period(3.82, 4.264, 0.0).expect("lattice constants are positive");
    let glue = SfGlueConfig {
        eta: 0.1,
        tc_target: 5.0,
        coherence: SfCoherence::Clean,
        moire_period: period,
        ..SfGlueConfig::default()
    };
    let map = glue_modulated_gap(&moire.pattern, &glue)
        .expect("the glue map must compute for FeTe / Sb2Te3 at 0°");
    let norm = normalize(&map.gap).expect("dressed gap must be normalizable");
    render_surface_3d_opts(
        &norm,
        moire.resolution,
        SIZE,
        SIZE,
        &tilt_camera(),
        coolwarm,
        bg_dark(),
        &default_opts(MESH_WINDOW),
    )
}

/// Moire-locked PDW what-if above Tc (FeTe / Sb2Te3, T = 16 K, T_PDW =
/// 1.5 Tc), mirroring the web `pdw-moire-whatif` scene: the pair amplitude
/// |d(r)| as a surface (220² over 200 Å, coolwarm, tilt camera). The |d|
/// maxima stand up as red domes on a triangular lattice, 42.4 Å apart
/// (2L_m/√3); each sits in a blue trough of nodes (|d| = 0), and pale
/// saddles join the troughs into a honeycomb. Amplitudes follow the desktop
/// app's moire-locked preset.
fn scene_rust_pdw_whatif_3d() -> ColorImage {
    let extent = 200.0_f64;
    let period = moire_locked_period(3.82, 4.264, 0.0).expect("lattice constants are positive");
    let density = DensityConfig::default();
    let ratio = moire_locked_pdw_ratio(period, 20.0, density.delta_1, density.delta_2)
        .expect("FeTe / Sb2Te3 has a finite moire period");
    let delta_avg = (density.delta_1 + density.delta_2) / 2.0;
    let base = PdwConfig::moire_locked(period, 0.0, delta_avg, ratio)
        .expect("FeTe / Sb2Te3 has a finite moire period");
    let cfg = PdwConfig {
        temperature: 16.0,
        t_pdw: 1.5 * TC_FETE_FILM,
        ..base
    };
    let state = compute_pdw_state(&cfg).expect("PDW state must compute for the what-if preset");
    let fields = pdw_real_space(&cfg, &state, 220, extent).expect("PDW fields must compute");
    let norm = normalize(&fields.pair_amplitude).expect("pair amplitude must be normalizable");
    render_surface_3d_opts(
        &norm,
        fields.resolution,
        SIZE,
        SIZE,
        &tilt_camera(),
        coolwarm,
        bg_dark(),
        &default_opts(extent),
    )
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

fn usage() -> String {
    let mut text = String::from(
        "usage: cargo run --release -p moire-desktop --bin capture -- \
         [--list] [--only NAME[,NAME...]] [--out-dir DIR]\n\n\
         Renders the desktop-app docs figures as 1024x1024 PNGs.\n\n\
         options:\n  \
         --list          print name, file, frame, kind and size for each scene\n  \
         --only NAMES    render only these scenes (comma list; may repeat;\n                  \
         an unknown name is an error)\n  \
         --out-dir DIR   output directory (default: the repository's docs/images)\n\n\
         scenes:\n",
    );
    for scene in SCENES {
        text.push_str(&format!(
            "  {:<24} {:<14} {}\n",
            scene.name, scene.kind, scene.filename
        ));
    }
    text
}

/// Size in bytes of every `.png` directly inside `dir`.
fn png_sizes(dir: &Path) -> Result<Vec<u64>, String> {
    let mut sizes = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| format!("read {}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|ext| ext == "png") {
            let meta = fs::metadata(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            sizes.push(meta.len());
        }
    }
    Ok(sizes)
}

/// PNG row filters `save_png` tries, in tie-break order. Flat-shaded 3D
/// renders compress best unfiltered; smooth 2D textures prefer Sub or Paeth.
const PNG_FILTERS: [png::FilterType; 5] = [
    png::FilterType::NoFilter,
    png::FilterType::Sub,
    png::FilterType::Up,
    png::FilterType::Avg,
    png::FilterType::Paeth,
];

/// Encode 8-bit RGB pixels as a PNG in memory, with one filter for every row.
fn encode_png(
    rgb: &[u8],
    width: usize,
    height: usize,
    filter: png::FilterType,
) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    let mut encoder = png::Encoder::new(&mut bytes, width as u32, height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Best);
    encoder.set_filter(filter);
    let mut writer = encoder
        .write_header()
        .map_err(|e| format!("png header: {e}"))?;
    writer
        .write_image_data(rgb)
        .map_err(|e| format!("png data: {e}"))?;
    writer.finish().map_err(|e| format!("png end: {e}"))?;
    Ok(bytes)
}

/// Write an opaque capture as an 8-bit RGB PNG. Unlike the app's screenshot
/// writer (`render::screenshot`: RGBA with png's default filter and level),
/// it drops the constant alpha channel, compresses at png's `Best` level and
/// keeps the smallest of the five row filters, which roughly halves the
/// folder weight. The bytes are deterministic for a fixed `png` crate version.
fn save_png(image: &ColorImage, path: &Path) -> Result<(), String> {
    let [width, height] = image.size;
    let mut rgb = Vec::with_capacity(width * height * 3);
    for pixel in &image.pixels {
        if pixel.a() != 255 {
            return Err(format!(
                "{}: capture has a transparent pixel",
                path.display()
            ));
        }
        rgb.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b()]);
    }
    let encode = |filter| {
        encode_png(&rgb, width, height, filter).map_err(|e| format!("{}: {e}", path.display()))
    };
    let mut smallest = encode(PNG_FILTERS[0])?;
    for &filter in &PNG_FILTERS[1..] {
        let candidate = encode(filter)?;
        if candidate.len() < smallest.len() {
            smallest = candidate;
        }
    }
    fs::write(path, smallest).map_err(|e| format!("write {}: {e}", path.display()))
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("capture: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), String> {
    let mut out_dir = PathBuf::from(DEFAULT_OUT_DIR);
    let mut only: Vec<String> = Vec::new();
    let mut list = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--out-dir" => {
                out_dir = args
                    .next()
                    .map(PathBuf::from)
                    .ok_or("--out-dir needs a path")?;
            }
            "--only" => {
                let raw = args.next().ok_or("--only needs NAME[,NAME...]")?;
                let before = only.len();
                only.extend(
                    raw.split(',')
                        .map(str::trim)
                        .filter(|name| !name.is_empty())
                        .map(String::from),
                );
                if only.len() == before {
                    return Err("--only needs at least one scene name".to_string());
                }
            }
            "--list" => list = true,
            "--help" | "-h" => {
                print!("{}", usage());
                return Ok(());
            }
            other => return Err(format!("unknown argument '{other}'\n\n{}", usage())),
        }
    }

    if let Some(bad) = only
        .iter()
        .find(|name| !SCENES.iter().any(|s| s.name == name.as_str()))
    {
        let known: Vec<&str> = SCENES.iter().map(|s| s.name).collect();
        return Err(format!(
            "unknown scene '{bad}'; known scenes: {} (see --list)",
            known.join(", ")
        ));
    }
    let selected: Vec<&Scene> = SCENES
        .iter()
        .filter(|s| only.is_empty() || only.iter().any(|name| name == s.name))
        .collect();

    if list {
        for scene in &selected {
            println!(
                "{}\t{}\t{FRAME}\t{}\t{SIZE}x{SIZE}",
                scene.name, scene.filename, scene.kind
            );
        }
        return Ok(());
    }

    fs::create_dir_all(&out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;
    let out_dir = fs::canonicalize(&out_dir).unwrap_or(out_dir);

    let mut written = 0_u64;
    for scene in &selected {
        eprintln!("[{}] rendering…", scene.name);
        let start = Instant::now();
        let image = (scene.build)();
        let path = out_dir.join(scene.filename);
        save_png(&image, &path)?;
        let bytes = fs::metadata(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len();
        written += bytes;
        println!(
            "[{}] wrote {} ({} KB, {:.1} s)",
            scene.name,
            path.display(),
            bytes / 1024,
            start.elapsed().as_secs_f64()
        );
        if bytes > SOFT_CAP_BYTES {
            println!(
                "[{}] warning: above the {} KB soft cap per PNG",
                scene.name,
                SOFT_CAP_BYTES / 1024
            );
        }
    }

    let folder = png_sizes(&out_dir)?;
    let total: u64 = folder.iter().sum();
    println!(
        "wrote {} PNGs, {} KB; {} now holds {} PNGs, {} KB of the {} KB budget",
        selected.len(),
        written / 1024,
        out_dir.display(),
        folder.len(),
        total / 1024,
        FOLDER_BUDGET_BYTES / 1024
    );
    if total > FOLDER_BUDGET_BYTES {
        println!("warning: the folder is over the 12 MB docs figure budget");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_files_and_kinds_are_consistent() {
        for (i, scene) in SCENES.iter().enumerate() {
            assert_eq!(scene.filename, format!("{}.png", scene.name));
            assert!(scene.name.starts_with("rust-"));
            assert!(matches!(scene.kind, "app" | "presentation"));
            assert!(SCENES[..i].iter().all(|s| s.name != scene.name));
        }
    }

    #[test]
    fn srgb_round_trip_is_exact_for_every_level() {
        for c in 0..=255_u8 {
            assert_eq!(srgb_from_linear(linear_from_srgb(c)), c);
        }
    }

    #[test]
    fn texture_puts_row_zero_at_the_bottom_and_magnifies_bilinearly() {
        // 2 x 2 field: data row 0 is 0.0, row 1 is 1.0; row 0 (y_min) is
        // drawn at the bottom, as the app's v-flipped texture shows it.
        let data = [0.0, 0.0, 1.0, 1.0];
        let gray = |t: f64| {
            let v = (t * 255.0).round() as u8;
            [v, v, v, 255]
        };
        let img = render_texture_2d(&data, 2, 8, gray);
        assert_eq!(img.size, [8, 8]);
        // Edge pixels clamp to their texel; the middle rows blend.
        assert_eq!(img.pixels[0], Color32::from_gray(255));
        assert_eq!(img.pixels[7 * 8], Color32::from_gray(0));
        let middle = img.pixels[3 * 8].r();
        assert!(middle > 0 && middle < 255);
        // n == size reproduces the colormap texel for texel.
        let same = render_texture_2d(&data, 2, 2, gray);
        assert_eq!(same.pixels[0], Color32::from_gray(255));
        assert_eq!(same.pixels[2], Color32::from_gray(0));
    }

    #[test]
    fn interior_blur_keeps_constants_and_trims_the_pad() {
        let sigma = 2.0;
        let r = blur_radius(sigma);
        let n = 4 * r;
        let field: Vec<f64> = (0..n * n).map(|_| 0.7).collect();
        let out = gaussian_blur_interior(&field, n, sigma);
        assert_eq!(out.len(), (n - 2 * r) * (n - 2 * r));
        assert!(out.iter().all(|v| (v - 0.7).abs() < 1e-12));
    }

    #[test]
    fn sf_glue_scene_is_the_app_capture() {
        use moire_desktop::app::{MoireApp, SfSettings, SfView, Tab, ViewMode};

        // The running app at the scene's settings: desktop defaults (FeTe /
        // Sb₂Te₃, θ = 0, 256²) in the 50 Å window, the Spin Fluctuations tab's
        // Glue gap view in 3D, Tc target 5 K, the tilt camera.
        let mut app = MoireApp {
            physical_extent: MESH_WINDOW,
            view_mode: ViewMode::Surface3D,
            active_tab: Tab::SpinFluctuation,
            sf_view: SfView::GlueGap,
            sf_settings: SfSettings {
                tc_target: 5.0,
                ..SfSettings::default()
            },
            camera: tilt_camera(),
            ..MoireApp::default()
        };
        app.run_compute_stages(&egui::Context::default());
        let captured = app
            .capture_current_view()
            .expect("the glue map captures in 3D");
        let scene = scene_rust_sf_glue_3d();
        assert_eq!(captured.size, scene.size);
        assert!(
            captured.pixels == scene.pixels,
            "rust-sf-glue-3d drifted from the app's own capture"
        );
    }

    #[test]
    fn resample_is_corner_aligned() {
        let n = 5;
        let data: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let out = resample_bilinear(&data, n, 9);
        let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
        assert!(close(out[0], 0.0));
        assert!(close(out[8], 4.0));
        assert!(close(out[9 * 9 - 1], 24.0));
        assert!(close(out[1], 0.5));
    }
}
