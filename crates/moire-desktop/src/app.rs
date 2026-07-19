//! `MoireApp` — the egui/eframe application state machine.
//!
//! Recompute-on-change: user mutations set `needs_recompute` / `needs_*`
//! flags, and the next `update()` tick runs the pipeline (isotope → moire →
//! density → FFT → textures → 3D re-render). Menu bar and keyboard shortcuts
//! route through `MenuAction` in `ui::menu`; the 2D viewport reads axis and
//! colorbar metadata from `ui::viewport::view_meta`.

use eframe::egui;
use moire_core::bm_model::{BMConfig, BandStructure, DosResult};
use moire_core::colormap::ColormapName;
use moire_core::curvature::{CurvatureConfig, CurvatureGeometry, CurvatureResult};
use moire_core::density::{DensityConfig, DensityResult};
use moire_core::graphene::{
    FlatBandConfig, GrapheneStackConfig, GrapheneStackConfigV2, GrapheneStackResult, StackingKind,
    SupermoireConfig, SupermoireResult,
};
use moire_core::isotope_effects::{IsotopeEffects, IsotopeEffectsConfig};
use moire_core::isotopes::IsotopeConfig;
use moire_core::magnetic::{MagneticFieldConfig, VortexLatticeResult, ZeemanResult};
use moire_core::materials;
use moire_core::moire::{MoireConfig, MoireResult};
use moire_core::topological::ProximityConfig;

use crate::render;
use crate::render::renderer3d::Renderer3D;
use crate::ui;

/// Default BCS coherence length for FeTe (Angstrom).
const DEFAULT_COHERENCE_LENGTH: f64 = 20.0;

/// Default Majorana localization length (Angstrom); mirrors Python
/// `XI_MAJORANA_DEFAULT` in config.py.
const DEFAULT_MAJORANA_XI: f64 = 50.0;
/// Default TI surface Fermi wavevector (1/Angstrom); mirrors Python `K_F_TSS`.
const DEFAULT_KF: f64 = 0.1;

/// London penetration depth for FeTe (Angstrom); mirrors Python `LAMBDA_L_FETE`.
const DEFAULT_LAMBDA_L: f64 = 5000.0;

/// FFT peak-detection threshold, as a fraction of the (log-normalized) peak.
const FFT_PEAK_THRESHOLD: f64 = 0.5;
/// Maximum FFT peaks retained for the table (matches the web page).
const FFT_MAX_PEAKS: usize = 20;

/// k-points per segment of the K -> Gamma -> M -> K' path (Bands view).
const BM_N_K_PER_SEGMENT: usize = 12;
/// k-grid subdivisions per moire reciprocal vector (DOS view).
const BM_N_K_GRID: usize = 10;
/// Half-width of the DOS energy window in meV.
const DOS_E_WINDOW_MEV: f64 = 150.0;
/// Number of DOS histogram bins.
const DOS_N_BINS: usize = 200;
/// Gaussian DOS broadening in meV.
const DOS_BROADENING_MEV: f64 = 2.0;

/// Owned scalar field + resolution + colormap backing a view texture.
type ViewScalar = (Vec<f64>, usize, fn(f64) -> [u8; 4]);

/// Per-tab data feeding one 3D surface frame. `heights` drives vertex
/// displacement; `colors` (when `Some`) drives the colormap independently —
/// the curved-sheet path where geometry and color come from different fields.
/// `None` colors by height.
struct SurfaceScalar {
    heights: Vec<f64>,
    colors: Option<Vec<f64>>,
    n: usize,
    colormap: fn(f64) -> [u8; 4],
}

/// Normalize a slice of f64 values to [0, 1] range.
/// Returns None if the range is too small (< 1e-15).
fn normalize_to_unit_range(data: &[f64]) -> Option<Vec<f64>> {
    let min_val = data.iter().copied().fold(f64::MAX, f64::min);
    let max_val = data.iter().copied().fold(f64::MIN, f64::max);
    let range = max_val - min_val;
    if range < 1e-15 {
        return None;
    }
    Some(data.iter().map(|&v| (v - min_val) / range).collect())
}

/// Global value range (meV) of the separable 3D gap field `gap_2d * decay[z]`.
/// Because the field is a product, the extremes are among the four products of
/// the gap and decay-profile extremes. Used to normalize z-slices against the
/// whole volume so deeper slices visibly dim instead of renormalizing per slice.
fn cooper_global_range(combined: &[f64], decay: &[f64]) -> (f64, f64) {
    let g_min = combined.iter().copied().fold(f64::INFINITY, f64::min);
    let g_max = combined.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let d_min = decay.iter().copied().fold(f64::INFINITY, f64::min);
    let d_max = decay.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let products = [g_min * d_min, g_min * d_max, g_max * d_min, g_max * d_max];
    let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (lo, hi)
}

/// Active visualization tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Pattern,
    Density,
    Fourier,
    MagneticField,
    CooperSurface3D,
    Graphene,
}

/// 2D flat or 3D surface view mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ViewMode {
    Flat2D,
    Surface3D,
}

/// Which scalar field or plot the Graphene tab displays. `Bands` and `Dos`
/// render as egui_plot line charts rather than colormapped textures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum GrapheneView {
    #[default]
    Pattern,
    GapMap,
    PseudoField,
    Strain,
    Fourier,
    Bands,
    Dos,
}

/// Which scalar field or plot the Cooper 3D / proximity tab displays.
/// `DecayProfile` renders as an egui_plot line chart rather than a texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum CooperView {
    /// Gap at the first interface layer (z >= 0): T * Delta(r).
    #[default]
    InterfaceGap,
    /// gap_2d * decay[z_slice_index], normalized against the global 3D range.
    ZSlice,
    /// f(z) decay profile as a line plot.
    DecayProfile,
    /// SPECULATIVE vortex-bound Majorana probability density at the z-slice.
    Majorana,
}

/// Which scalar field the Magnetic tab displays. Susceptibility and screening
/// currents are 2D magnitude maps (no 3D cones, unlike the web page).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum MagneticView {
    /// Gap suppressed by the vortex lattice, Δ(r) · S(r).
    #[default]
    CombinedGap,
    /// SPECULATIVE local magnetic susceptibility χ(r), mapped to [0, 1].
    Susceptibility,
    /// Meissner screening supercurrent magnitude |j|(r), peak-normalized.
    ScreeningCurrent,
}

impl Default for Tab {
    fn default() -> Self {
        Tab::Pattern
    }
}

impl Default for ViewMode {
    fn default() -> Self {
        ViewMode::Flat2D
    }
}

/// Main application state.
#[derive(serde::Deserialize, serde::Serialize)]
pub struct MoireApp {
    /// Index into the materials list for the substrate.
    pub substrate_idx: usize,
    /// Index into the materials list for the overlayer.
    pub overlayer_idx: usize,
    /// Twist angle of the overlayer in degrees.
    pub twist_angle: f64,
    /// Grid resolution (NxN).
    pub resolution: usize,
    /// Viewport size in Angstroms.
    pub physical_extent: f64,
    /// Density modulation parameters.
    #[serde(with = "density_config_serde")]
    pub density_config: DensityConfig,
    /// View mode (2D or 3D).
    pub view_mode: ViewMode,

    // --- Isotope enrichment (speculative) ---
    /// Whether isotope effects are enabled.
    #[serde(default)]
    pub isotope_enabled: bool,
    /// Fe effective mass override (amu), None = natural average.
    #[serde(default)]
    pub fe_mass_override: Option<f64>,
    /// Te effective mass override (amu).
    #[serde(default)]
    pub te_mass_override: Option<f64>,
    /// Sb effective mass override (amu).
    #[serde(default)]
    pub sb_mass_override: Option<f64>,
    /// C effective mass override (amu) — used when graphene is selected.
    #[serde(default)]
    pub c_mass_override: Option<f64>,
    /// BCS isotope exponent.
    #[serde(default = "default_isotope_alpha")]
    pub isotope_alpha: f64,
    /// Whether the HIGHLY SPECULATIVE exotic-isotope mass ranges are active.
    #[serde(default)]
    pub exotic_mode: bool,

    // --- Magnetic field parameters ---
    /// Magnetic field configuration.
    #[serde(default)]
    pub magnetic_config: MagneticFieldConfig,
    /// Proximity effect configuration.
    #[serde(default)]
    pub proximity_config: ProximityConfig,
    /// Effective g-factor.
    #[serde(default = "default_g_factor")]
    pub g_factor: f64,
    /// Whether to show vortex core markers.
    #[serde(default)]
    pub show_vortices: bool,
    /// Whether to show Majorana density (speculative).
    #[serde(default)]
    pub show_majorana: bool,

    /// Whether dark mode is enabled.
    #[serde(default = "default_dark_mode")]
    pub dark_mode: bool,

    /// Whether to render the 3D surface as wireframe instead of filled.
    #[serde(default)]
    pub show_wireframe: bool,
    /// Whether to draw world axes and a scale bar on the 3D surface.
    #[serde(default = "default_true")]
    pub show_world_axes: bool,
    /// Whether the clipping plane is active. When off, the full mesh renders.
    #[serde(default)]
    pub clip_z_enabled: bool,
    /// Clipping plane height in normalised surface coordinates (-1..1).
    /// Consumed by the wgpu shader's `uniforms.clip_z` and — once the CPU
    /// rasterizer gains a clip pass — by the software path too.
    #[serde(default = "default_clip_z")]
    pub clip_z: f32,

    // --- Graphene stack (speculative) ---
    /// Stacking arrangement for the graphene tab.
    #[serde(default)]
    pub graphene_stack: StackingKind,
    /// Twist angle for twisted stackings (degrees).
    #[serde(default = "default_graphene_twist")]
    pub graphene_twist: f64,
    /// Moire flat-band filling |nu|.
    #[serde(default = "default_graphene_filling")]
    pub graphene_filling: f64,
    /// Sheet curvature configuration (speculative).
    #[serde(default)]
    pub curvature_config: CurvatureConfig,
    /// Which scalar field the graphene tab shows.
    #[serde(default)]
    pub graphene_view: GrapheneView,
    /// Include the B-sublattice (honeycomb) term in each layer potential.
    #[serde(default)]
    pub graphene_honeycomb: bool,
    /// Valley index: +1 = K, -1 = K'.
    #[serde(default = "default_valley")]
    pub graphene_valley: i32,
    /// Uniaxial heterostrain on layer index 1, in percent.
    #[serde(default)]
    pub graphene_strain_percent: f64,
    /// Heterostrain tension axis, in degrees.
    #[serde(default)]
    pub graphene_strain_angle: f64,
    /// Warp the stack sampling by the curvature displacement field.
    #[serde(default)]
    pub graphene_warp: bool,
    /// Supermoire overlayer formula (None = plain graphene stack).
    #[serde(default)]
    pub supermoire_overlayer: Option<String>,
    /// Twist between the graphene stack and the supermoire overlayer (deg).
    #[serde(default)]
    pub supermoire_interface_twist: f64,

    // --- Cooper 3D / proximity ---
    /// Which scalar field or plot the Cooper 3D tab displays.
    #[serde(default)]
    pub cooper_view: CooperView,

    /// Global colormap override. `None` = Auto (each view keeps its semantic
    /// palette: viridis unsigned, coolwarm signed, inferno FFT, plasma χ/|j|).
    #[serde(default)]
    pub colormap_override: Option<ColormapName>,

    /// Which scalar field the Magnetic tab displays.
    #[serde(default)]
    pub magnetic_view: MagneticView,

    // --- Topological phase diagram (speculative) ---
    /// Max perpendicular field B on the phase-diagram B axis (Tesla).
    #[serde(default = "default_phase_b_max")]
    pub phase_b_max: f64,
    /// Max gap Δ on the phase-diagram Δ axis (meV).
    #[serde(default = "default_phase_delta_max")]
    pub phase_delta_max: f64,
    /// Chemical potential μ used by the Fu-Kane phase criterion (meV).
    #[serde(default)]
    pub phase_mu: f64,

    // --- Runtime state (not serialized) ---
    /// Currently selected tab.
    #[serde(skip)]
    pub active_tab: Tab,
    /// Whether to recompute on next frame.
    #[serde(skip)]
    pub needs_recompute: bool,
    /// Whether the 3D surface needs re-rendering (camera or data changed).
    #[serde(skip)]
    pub needs_surface_rerender: bool,
    /// Cached moire result.
    #[serde(skip)]
    pub moire_result: Option<MoireResult>,
    /// Cached density result.
    #[serde(skip)]
    pub density_result: Option<DensityResult>,
    /// FFT power spectrum data.
    #[serde(skip)]
    pub fft_data: Option<Vec<f64>>,
    /// Detected FFT peaks (top `FFT_MAX_PEAKS`, amplitude-descending).
    #[serde(skip)]
    pub fft_peaks: Option<Vec<moire_core::fft::FftPeak>>,
    /// Texture for the moire pattern.
    #[serde(skip)]
    pub pattern_texture: Option<egui::TextureHandle>,
    /// Texture for the density modulation.
    #[serde(skip)]
    pub density_texture: Option<egui::TextureHandle>,
    /// Texture for the FFT power spectrum.
    #[serde(skip)]
    pub fft_texture: Option<egui::TextureHandle>,
    /// 3D surface camera.
    #[serde(skip)]
    pub camera: render::surface3d::Camera3D,
    /// Rendered 3D surface texture.
    #[serde(skip)]
    pub surface_texture: Option<egui::TextureHandle>,
    /// Active 3D renderer backend. Lazily initialized to the software
    /// rasterizer; a GPU backend can be swapped in behind the `gpu` feature
    /// without touching the per-tab surface pipeline.
    #[serde(skip)]
    pub renderer: Option<Box<dyn render::renderer3d::Renderer3D>>,
    /// Whether the comparison window is open.
    #[serde(skip)]
    pub show_comparison: bool,
    /// Whether the topological phase-diagram window is open.
    #[serde(skip)]
    pub show_phase_diagram: bool,
    /// Texture for the phase-diagram image.
    #[serde(skip)]
    pub phase_texture: Option<egui::TextureHandle>,
    /// Whether the phase-diagram texture needs refresh.
    #[serde(skip)]
    pub phase_needs_refresh: bool,
    /// Textures for substrate comparison (6: 3 moire + 3 density).
    #[serde(skip)]
    pub comparison_textures: Option<Vec<egui::TextureHandle>>,
    /// Whether comparison textures need refresh.
    #[serde(skip)]
    pub comparison_needs_refresh: bool,
    /// Cached isotope effects (when enabled).
    #[serde(skip)]
    pub isotope_effects: Option<IsotopeEffects>,
    /// Cached vortex lattice result.
    #[serde(skip)]
    pub vortex_result: Option<VortexLatticeResult>,
    /// Cached Zeeman result.
    #[serde(skip)]
    pub zeeman_result: Option<ZeemanResult>,
    /// Whether magnetic effects need recompute.
    #[serde(skip)]
    pub needs_magnetic_recompute: bool,
    /// Texture for the magnetic field visualization.
    #[serde(skip)]
    pub magnetic_texture: Option<egui::TextureHandle>,
    /// Screening-current magnitude |j|(r), peak-normalized; computed lazily
    /// only while the ScreeningCurrent view is active.
    #[serde(skip)]
    pub screening_field: Option<Vec<f64>>,
    /// Z-slice index for proximity 3D view.
    #[serde(skip)]
    pub z_slice_index: usize,
    /// Whether the About dialog is currently shown.
    #[serde(skip)]
    pub show_about: bool,
    /// Last screenshot status message (shown transiently in the info panel).
    #[serde(skip)]
    pub last_screenshot_status: Option<String>,
    /// Whether the graphene stack needs recompute.
    #[serde(skip)]
    pub needs_graphene_recompute: bool,
    /// Cached graphene stack result.
    #[serde(skip)]
    pub graphene_result: Option<GrapheneStackResult>,
    /// Cached sheet curvature result.
    #[serde(skip)]
    pub curvature_result: Option<CurvatureResult>,
    /// Curvature-suppressed flat-band gap field (meV).
    #[serde(skip)]
    pub graphene_gap: Option<Vec<f64>>,
    /// Texture for the active graphene view.
    #[serde(skip)]
    pub graphene_texture: Option<egui::TextureHandle>,
    /// FFT power spectrum of the graphene pattern (Fourier view).
    #[serde(skip)]
    pub graphene_fft: Option<Vec<f64>>,
    /// Cached supermoire result (when an overlayer is active).
    #[serde(skip)]
    pub supermoire_result: Option<SupermoireResult>,
    /// Cached BM band structure (Bands view; computed lazily).
    #[serde(skip)]
    pub band_structure: Option<BandStructure>,
    /// Cached BM density of states (DOS view; computed lazily).
    #[serde(skip)]
    pub dos_result: Option<DosResult>,
    /// (effective twist, valley) the cached BM results were computed for.
    #[serde(skip)]
    pub bm_cache_key: Option<(f64, i32)>,

    // --- Cooper 3D runtime state (not serialized) ---
    /// Combined 2D gap incl. vortex suppression (meV), the z=0 base field.
    #[serde(skip)]
    pub cooper_gap: Option<Vec<f64>>,
    /// z-coordinate grid (Angstrom) for the proximity volume.
    #[serde(skip)]
    pub cooper_z_coords: Option<Vec<f64>>,
    /// 1D proximity decay profile f(z).
    #[serde(skip)]
    pub cooper_decay: Option<Vec<f64>>,
    /// Texture for the active Cooper view.
    #[serde(skip)]
    pub cooper_texture: Option<egui::TextureHandle>,
    /// Cached 3D Majorana probability volume (row-major nz*n*n), computed
    /// lazily and invalidated on each magnetic recompute.
    #[serde(skip)]
    pub majorana_density: Option<Vec<f64>>,
    /// Whether the Cooper stage needs recompute.
    #[serde(skip)]
    pub needs_cooper_recompute: bool,
    /// Whether the z-slice sweep animation is playing.
    #[serde(skip)]
    pub cooper_playing: bool,
    /// Time (seconds) of the last z-sweep advance.
    #[serde(skip)]
    pub cooper_last_tick: f64,
}

fn default_dark_mode() -> bool {
    true
}

fn default_true() -> bool {
    true
}

fn default_clip_z() -> f32 {
    1.0
}

fn default_isotope_alpha() -> f64 {
    0.25
}

fn default_g_factor() -> f64 {
    30.0
}

fn default_graphene_twist() -> f64 {
    1.08
}

fn default_graphene_filling() -> f64 {
    2.4
}

fn default_valley() -> i32 {
    1
}

fn default_phase_b_max() -> f64 {
    100.0
}

fn default_phase_delta_max() -> f64 {
    10.0
}

// Custom Serialize/Deserialize for DensityConfig so the outer derive works.
mod density_config_serde {
    use moire_core::density::DensityConfig;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    struct DensityConfigProxy {
        delta_1: f64,
        delta_2: f64,
        modulation_amplitude: f64,
        phase_shift: f64,
    }

    pub fn serialize<S: Serializer>(config: &DensityConfig, s: S) -> Result<S::Ok, S::Error> {
        let proxy = DensityConfigProxy {
            delta_1: config.delta_1,
            delta_2: config.delta_2,
            modulation_amplitude: config.modulation_amplitude,
            phase_shift: config.phase_shift,
        };
        proxy.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<DensityConfig, D::Error> {
        let proxy = DensityConfigProxy::deserialize(d)?;
        Ok(DensityConfig {
            delta_1: proxy.delta_1,
            delta_2: proxy.delta_2,
            modulation_amplitude: proxy.modulation_amplitude,
            phase_shift: proxy.phase_shift,
        })
    }
}

impl Default for MoireApp {
    fn default() -> Self {
        Self {
            substrate_idx: 0,
            overlayer_idx: 0,
            twist_angle: 0.0,
            resolution: 256,
            physical_extent: 200.0,
            density_config: DensityConfig::default(),
            view_mode: ViewMode::Flat2D,
            isotope_enabled: false,
            fe_mass_override: None,
            te_mass_override: None,
            sb_mass_override: None,
            c_mass_override: None,
            isotope_alpha: 0.4,
            exotic_mode: false,
            magnetic_config: MagneticFieldConfig::default(),
            proximity_config: ProximityConfig::default(),
            g_factor: 30.0,
            show_vortices: false,
            show_majorana: false,
            dark_mode: true,
            show_wireframe: false,
            show_world_axes: true,
            clip_z_enabled: false,
            clip_z: 1.0,
            graphene_stack: StackingKind::default(),
            graphene_twist: 1.08,
            graphene_filling: 2.4,
            curvature_config: CurvatureConfig::default(),
            graphene_view: GrapheneView::default(),
            graphene_honeycomb: false,
            graphene_valley: 1,
            graphene_strain_percent: 0.0,
            graphene_strain_angle: 0.0,
            graphene_warp: false,
            supermoire_overlayer: None,
            supermoire_interface_twist: 0.0,
            cooper_view: CooperView::default(),
            colormap_override: None,
            magnetic_view: MagneticView::default(),
            phase_b_max: 100.0,
            phase_delta_max: 10.0,
            phase_mu: 0.0,
            active_tab: Tab::Pattern,
            needs_recompute: true,
            needs_surface_rerender: true,
            moire_result: None,
            density_result: None,
            fft_data: None,
            pattern_texture: None,
            density_texture: None,
            fft_texture: None,
            fft_peaks: None,
            camera: render::surface3d::Camera3D::default(),
            surface_texture: None,
            renderer: None,
            show_comparison: false,
            show_phase_diagram: false,
            phase_texture: None,
            phase_needs_refresh: false,
            comparison_textures: None,
            comparison_needs_refresh: false,
            isotope_effects: None,
            vortex_result: None,
            zeeman_result: None,
            needs_magnetic_recompute: true,
            magnetic_texture: None,
            screening_field: None,
            z_slice_index: 0,
            show_about: false,
            last_screenshot_status: None,
            needs_graphene_recompute: true,
            graphene_result: None,
            curvature_result: None,
            graphene_gap: None,
            graphene_texture: None,
            graphene_fft: None,
            supermoire_result: None,
            band_structure: None,
            dos_result: None,
            bm_cache_key: None,
            cooper_gap: None,
            cooper_z_coords: None,
            cooper_decay: None,
            cooper_texture: None,
            majorana_density: None,
            needs_cooper_recompute: true,
            cooper_playing: false,
            cooper_last_tick: 0.0,
        }
    }
}

impl MoireApp {
    /// Create the application, restoring persisted state if available.
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut app = if let Some(storage) = cc.storage {
            eframe::get_value(storage, eframe::APP_KEY).unwrap_or_default()
        } else {
            Self::default()
        };
        // Apply persisted theme
        if app.dark_mode {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
        } else {
            cc.egui_ctx.set_visuals(egui::Visuals::light());
        }

        // Always recompute on startup since textures are not persisted
        app.needs_recompute = true;
        app.needs_surface_rerender = true;
        app
    }

    /// Get the substrate material based on current selection.
    pub fn substrate_material(&self) -> &'static moire_core::materials::Material {
        let substrates = materials::substrates();
        if substrates.is_empty() {
            // Should never happen — at least one substrate is always defined.
            return materials::substrate();
        }
        substrates[self.substrate_idx.min(substrates.len() - 1)]
    }

    /// Get the overlayer material based on current selection.
    pub fn overlayer_material(&self) -> &'static moire_core::materials::Material {
        let overlayers = materials::overlayers();
        if overlayers.is_empty() {
            // Fallback to substrate if no overlayers are defined
            return materials::substrate();
        }
        overlayers[self.overlayer_idx.min(overlayers.len() - 1)]
    }

    /// Resolve the persisted supermoire overlayer formula against the
    /// material database. Graphene-family entries are excluded: a graphene
    /// stack on graphene is just a larger stack, not a supermoire.
    pub fn supermoire_material(&self) -> Option<&'static moire_core::materials::Material> {
        let formula = self.supermoire_overlayer.as_deref()?;
        materials::overlayers()
            .into_iter()
            .find(|m| m.formula == formula && !m.formula.starts_with("Graphene"))
    }

    /// Build an IsotopeConfig from the current app state.
    fn isotope_config(&self) -> IsotopeConfig {
        IsotopeConfig {
            fe_mass: self.fe_mass_override,
            te_mass: self.te_mass_override,
            sb_mass: self.sb_mass_override,
            c_mass: self.c_mass_override,
        }
    }

    /// Run computation and create 2D textures.
    fn recompute(&mut self, ctx: &egui::Context) {
        let substrate = self.substrate_material();
        let overlayer = self.overlayer_material();

        let (sub_a, over_a, dw_sub, dw_over) = if self.isotope_enabled {
            let iso_cfg = IsotopeEffectsConfig {
                substrate_formula: substrate.formula,
                overlayer_formula: overlayer.formula,
                substrate_a: substrate.a,
                overlayer_a: overlayer.a,
                substrate_lattice_type: substrate.lattice_type,
                overlayer_lattice_type: overlayer.lattice_type,
                delta_1: self.density_config.delta_1,
                delta_2: self.density_config.delta_2,
                coherence_length: DEFAULT_COHERENCE_LENGTH,
                isotope_config: self.isotope_config(),
                alpha: self.isotope_alpha,
            };
            let effects = moire_core::isotope_effects::compute_isotope_effects(&iso_cfg);
            let result = (
                effects.substrate_a_modified,
                effects.overlayer_a_modified,
                effects.dw_factor_substrate,
                effects.dw_factor_overlayer,
            );
            self.isotope_effects = Some(effects);
            result
        } else {
            self.isotope_effects = None;
            (substrate.a, overlayer.a, 1.0, 1.0)
        };

        let config = MoireConfig {
            substrate_a: sub_a,
            substrate_lattice_type: substrate.lattice_type,
            overlayer_a: over_a,
            overlayer_lattice_type: overlayer.lattice_type,
            twist_angle_deg: self.twist_angle,
            resolution: self.resolution,
            physical_extent: self.physical_extent,
            dw_factor_substrate: dw_sub,
            dw_factor_overlayer: dw_over,
        };

        let moire = match moire_core::moire::compute_moire(&config) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Moire computation error: {e}");
                return;
            }
        };
        let density =
            match moire_core::density::compute_density_modulation(&moire, &self.density_config) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("Density computation error: {e}");
                    return;
                }
            };
        let fft_data = match moire_core::fft::compute_fft_2d(&moire.pattern, moire.resolution) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("FFT computation error: {e}");
                return;
            }
        };

        // Create 2D textures
        self.pattern_texture = Some(render::pattern::create_texture(
            ctx,
            "moire_pattern",
            &moire.pattern,
            moire.resolution,
            self.cmap(moire_core::colormap::viridis),
        ));

        // Normalize density for colormap
        let density_norm: Vec<f64> = normalize_to_unit_range(&density.gap_field)
            .unwrap_or_else(|| vec![0.5; density.gap_field.len()]);

        self.density_texture = Some(render::pattern::create_texture(
            ctx,
            "density_map",
            &density_norm,
            density.resolution,
            self.cmap(moire_core::colormap::coolwarm),
        ));

        self.fft_texture = Some(render::pattern::create_texture(
            ctx,
            "fft_spectrum",
            &fft_data,
            moire.resolution,
            self.cmap(moire_core::colormap::inferno),
        ));

        // Detect FFT peaks for the Fourier-tab table. The spectrum is
        // log-normalized to [0, 1], so the threshold is a fraction of its peak.
        let dx = moire.physical_extent / (moire.resolution.saturating_sub(1).max(1)) as f64;
        let k = moire_core::fft::fft_frequencies(moire.resolution, dx);
        let mut peaks =
            moire_core::fft::identify_peaks(&fft_data, moire.resolution, &k, &k, FFT_PEAK_THRESHOLD);
        peaks.truncate(FFT_MAX_PEAKS);
        self.fft_peaks = Some(peaks);

        self.fft_data = Some(fft_data);
        self.moire_result = Some(moire);
        self.density_result = Some(density);
        self.needs_surface_rerender = true;
        self.comparison_needs_refresh = true;
    }

    /// Dispatch a `MenuAction` (from either the menu bar or a keyboard
    /// shortcut) to the appropriate state mutation.
    fn apply_menu_action(&mut self, action: ui::menu::MenuAction, ctx: &egui::Context) {
        use ui::menu::MenuAction::*;
        match action {
            SaveScreenshot => self.save_screenshot(),
            ResetParameters => self.reset_parameters(),
            ResetCamera => {
                self.camera = render::surface3d::Camera3D::default();
                self.needs_surface_rerender = true;
            }
            ToggleWireframe => {
                self.show_wireframe = !self.show_wireframe;
                self.needs_surface_rerender = true;
            }
            ToggleAxes => {
                self.show_world_axes = !self.show_world_axes;
                self.needs_surface_rerender = true;
            }
            ToggleTheme => {
                self.dark_mode = !self.dark_mode;
                if self.dark_mode {
                    ctx.set_visuals(egui::Visuals::dark());
                } else {
                    ctx.set_visuals(egui::Visuals::light());
                }
                self.needs_surface_rerender = true;
            }
            ShowAbout => {
                self.show_about = !self.show_about;
            }
            Quit => {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }
    }

    /// Restore user-editable parameters (twist, extent, resolution, gaps,
    /// isotope and magnetic flags) to their defaults. Runtime caches and
    /// textures are cleared so the next frame recomputes everything.
    pub fn reset_parameters(&mut self) {
        *self = Self {
            dark_mode: self.dark_mode,
            ..Self::default()
        };
    }

    /// Build a high-resolution ColorImage of the currently-displayed view
    /// without touching on-screen textures.
    fn capture_current_view(&self) -> Option<egui::ColorImage> {
        use moire_core::colormap;

        let opts = self.surface_opts();
        let bg = if self.dark_mode {
            egui::Color32::from_rgb(30, 30, 35)
        } else {
            egui::Color32::from_rgb(240, 240, 245)
        };
        const CAPTURE_W: usize = 1024;
        const CAPTURE_H: usize = 1024;

        if self.view_mode == ViewMode::Surface3D {
            let scalar = self.surface_scalar()?;
            // A one-shot software render at capture resolution; the persistent
            // `self.renderer` is not borrowed here so `&self` is preserved.
            let mut renderer = render::renderer3d::SoftwareRenderer;
            return Some(renderer.render(render::renderer3d::FrameInputs {
                data: &scalar.heights,
                n: scalar.n,
                size: [CAPTURE_W, CAPTURE_H],
                camera: &self.camera,
                colormap: scalar.colormap,
                background: bg,
                opts: &opts,
                clip_z: self.clip_z_enabled.then_some(self.clip_z),
                color_data: scalar.colors.as_deref(),
            }));
        }

        let (data_vec, n, colormap): ViewScalar = match self.active_tab {
            Tab::Pattern => {
                let m = self.moire_result.as_ref()?;
                (m.pattern.clone(), m.resolution, self.cmap(colormap::viridis))
            }
            Tab::Density => {
                let d = self.density_result.as_ref()?;
                let norm = normalize_to_unit_range(&d.gap_field)?;
                (norm, d.resolution, self.cmap(colormap::coolwarm))
            }
            Tab::CooperSurface3D => {
                let (data, cmap) = self.cooper_view_scalar()?;
                let n = self.density_result.as_ref()?.resolution;
                (data, n, cmap)
            }
            Tab::Fourier => {
                let f = self.fft_data.as_ref()?;
                let m = self.moire_result.as_ref()?;
                (f.clone(), m.resolution, self.cmap(colormap::inferno))
            }
            Tab::MagneticField => {
                let (data, cmap) = self.magnetic_view_scalar()?;
                let n = self.density_result.as_ref()?.resolution;
                (data, n, cmap)
            }
            Tab::Graphene => {
                let (data, cmap) = self.graphene_view_scalar()?;
                let n = self.graphene_result.as_ref()?.resolution;
                (data, n, cmap)
            }
        };

        // 2D: colormap the grid directly at the source resolution, then let
        // the PNG encoder upscale perceptually via viewer software. (We keep
        // the raw grid resolution rather than interpolating to 1024² in CPU.)
        let pixels: Vec<egui::Color32> = data_vec
            .iter()
            .map(|&v| {
                let [r, g, b, a] = colormap(v);
                egui::Color32::from_rgba_premultiplied(r, g, b, a)
            })
            .collect();
        Some(egui::ColorImage {
            size: [n, n],
            pixels,
        })
    }

    /// Save the current view to a timestamped PNG in the working directory.
    /// Updates `last_screenshot_status` with the outcome.
    pub fn save_screenshot(&mut self) {
        let Some(image) = self.capture_current_view() else {
            let is_plot_view = (self.active_tab == Tab::Graphene
                && matches!(self.graphene_view, GrapheneView::Bands | GrapheneView::Dos))
                || (self.active_tab == Tab::CooperSurface3D
                    && self.cooper_view == CooperView::DecayProfile);
            let msg = if is_plot_view {
                "Screenshot skipped: band/DOS/decay plots are not captured"
            } else {
                "Screenshot skipped: nothing rendered yet"
            };
            self.last_screenshot_status = Some(msg.into());
            return;
        };
        let path = render::screenshot::default_screenshot_path();
        match render::screenshot::save_color_image_to_png(&image, &path) {
            Ok(()) => {
                self.last_screenshot_status = Some(format!("Saved {}", path.display()));
            }
            Err(e) => {
                self.last_screenshot_status = Some(format!("Screenshot failed: {e}"));
            }
        }
    }

    /// Resolve a view's semantic colormap against the global override. `None`
    /// override (Auto) keeps the semantic palette; `Some(name)` forces every
    /// view — and its colorbar — to that palette.
    pub fn cmap(&self, semantic: fn(f64) -> [u8; 4]) -> fn(f64) -> [u8; 4] {
        self.colormap_override
            .map(ColormapName::sample)
            .unwrap_or(semantic)
    }

    /// Options bundle that captures the current user toggles for 3D overlays.
    fn surface_opts(&self) -> render::surface3d::SurfaceRenderOpts {
        render::surface3d::SurfaceRenderOpts {
            show_wireframe: self.show_wireframe,
            show_axes: self.show_world_axes,
            show_scale_bar: self.show_world_axes,
            physical_extent: self.physical_extent as f32,
        }
    }

    /// Height / color / colormap for the active tab's 3D surface. `None` when
    /// the current view has no colormapped texture (e.g. Graphene Bands/Dos,
    /// Cooper Decay-profile) or its data is not yet computed. Shared by the
    /// on-screen re-render and the high-res screenshot capture so both stay in
    /// lockstep.
    fn surface_scalar(&self) -> Option<SurfaceScalar> {
        use moire_core::colormap;
        match self.active_tab {
            Tab::Pattern => {
                let m = self.moire_result.as_ref()?;
                Some(SurfaceScalar {
                    heights: m.pattern.clone(),
                    colors: None,
                    n: m.resolution,
                    colormap: self.cmap(colormap::viridis),
                })
            }
            Tab::Density => {
                let d = self.density_result.as_ref()?;
                let norm = normalize_to_unit_range(&d.gap_field)?;
                Some(SurfaceScalar {
                    heights: norm,
                    colors: None,
                    n: d.resolution,
                    colormap: self.cmap(colormap::coolwarm),
                })
            }
            Tab::CooperSurface3D => {
                // The z-slice / interface / Majorana scalar is already in
                // [0, 1]; height and color are the same field. DecayProfile
                // returns None (it renders as a line plot).
                let (data, cmap) = self.cooper_view_scalar()?;
                let n = self.density_result.as_ref()?.resolution;
                Some(SurfaceScalar {
                    heights: data,
                    colors: None,
                    n,
                    colormap: cmap,
                })
            }
            Tab::Fourier => {
                let f = self.fft_data.as_ref()?;
                let m = self.moire_result.as_ref()?;
                Some(SurfaceScalar {
                    heights: f.clone(),
                    colors: None,
                    n: m.resolution,
                    colormap: self.cmap(colormap::inferno),
                })
            }
            Tab::MagneticField => {
                // The 3D surface follows the active magnetic view (combined gap,
                // χ, or |j| as height); the picker recolors it.
                let (data, cmap) = self.magnetic_view_scalar()?;
                let n = self.density_result.as_ref()?.resolution;
                Some(SurfaceScalar {
                    heights: data,
                    colors: None,
                    n,
                    colormap: cmap,
                })
            }
            Tab::Graphene => {
                let (colors, cmap) = self.graphene_view_scalar()?;
                let n = self.graphene_result.as_ref()?.resolution;
                // Curved sheets displace by the height field and color by the
                // selected view scalar; flat sheets color by height directly.
                if self.curvature_config.geometry != CurvatureGeometry::Flat {
                    if let Some(ref c) = self.curvature_result {
                        if let Some(heights) = normalize_to_unit_range(&c.height) {
                            return Some(SurfaceScalar {
                                heights,
                                colors: Some(colors),
                                n,
                                colormap: cmap,
                            });
                        }
                    }
                }
                Some(SurfaceScalar {
                    heights: colors,
                    colors: None,
                    n,
                    colormap: cmap,
                })
            }
        }
    }

    /// Re-render the 3D surface texture from cached data through the active
    /// [`render::renderer3d::Renderer3D`] backend.
    fn rerender_surface(&mut self, ctx: &egui::Context) {
        let surface_bg = if self.dark_mode {
            egui::Color32::from_rgb(30, 30, 35)
        } else {
            egui::Color32::from_rgb(240, 240, 245)
        };
        let opts = self.surface_opts();
        let Some(scalar) = self.surface_scalar() else {
            return;
        };
        let clip_z = self.clip_z_enabled.then_some(self.clip_z);

        let renderer = self
            .renderer
            .get_or_insert_with(|| Box::new(render::renderer3d::SoftwareRenderer));
        let img = renderer.render(render::renderer3d::FrameInputs {
            data: &scalar.heights,
            n: scalar.n,
            size: [512, 512],
            camera: &self.camera,
            colormap: scalar.colormap,
            background: surface_bg,
            opts: &opts,
            clip_z,
            color_data: scalar.colors.as_deref(),
        });
        self.surface_texture =
            Some(ctx.load_texture("surface_3d", img, egui::TextureOptions::LINEAR));
    }

    /// Run magnetic field computation and create overlay texture.
    fn recompute_magnetic(&mut self, ctx: &egui::Context) {
        let moire_period = self
            .moire_result
            .as_ref()
            .map(|r| r.moire_period)
            .unwrap_or(f64::INFINITY);

        let resolution = self
            .moire_result
            .as_ref()
            .map(|r| r.resolution)
            .unwrap_or(self.resolution);

        let extent = self
            .moire_result
            .as_ref()
            .map(|r| r.physical_extent)
            .unwrap_or(self.physical_extent);

        // Compute vortex lattice + suppression
        let vortex = moire_core::magnetic::compute_magnetic_effects(
            &self.magnetic_config,
            moire_period,
            resolution,
            extent,
            DEFAULT_COHERENCE_LENGTH, // coherence length
        );
        let positions = vortex.vortex_positions.clone();
        // Set early so `magnetic_view_scalar` can read the suppression field.
        self.vortex_result = Some(vortex);

        // Screening currents are O(N² · n_vortices); compute the magnitude only
        // while its view is active and cache it (invalidated on every recompute).
        self.screening_field = if self.magnetic_view == MagneticView::ScreeningCurrent {
            let (jx, jy) =
                moire_core::magnetic::screening_currents(resolution, extent, &positions, DEFAULT_LAMBDA_L);
            Some(
                jx.iter()
                    .zip(&jy)
                    .map(|(&x, &y)| (x * x + y * y).sqrt())
                    .collect(),
            )
        } else {
            None
        };

        // Build the active-view texture.
        if let Some((data, cmap)) = self.magnetic_view_scalar() {
            let pixels: Vec<egui::Color32> = data
                .iter()
                .map(|&v| {
                    let [r, g, b, a] = cmap(v);
                    egui::Color32::from_rgba_premultiplied(r, g, b, a)
                })
                .collect();
            let mut img = egui::ColorImage {
                size: [resolution, resolution],
                pixels,
            };
            // Vortex core markers overlay every magnetic view when enabled.
            if self.show_vortices && !positions.is_empty() {
                let marker_color = if self.dark_mode {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                render::overlay::overlay_cross_markers(
                    &mut img,
                    &positions,
                    extent,
                    marker_color,
                    3,
                );
            }
            self.magnetic_texture =
                Some(ctx.load_texture("magnetic_overlay", img, egui::TextureOptions::LINEAR));
        }

        // Compute Zeeman (uses the app's g-factor, wired in Track 1).
        let delta_avg = (self.density_config.delta_1 + self.density_config.delta_2) / 2.0;
        self.zeeman_result = Some(moire_core::magnetic::compute_zeeman(
            &self.magnetic_config,
            delta_avg,
            self.g_factor,
        ));

        // The Cooper 3D field is built on the vortex suppression, so force it
        // to rebuild and drop the now-stale Majorana volume cache.
        self.needs_cooper_recompute = true;
        self.majorana_density = None;
        // Keep the 3D surface in step with the active magnetic view (mirrors
        // the other recompute stages).
        self.needs_surface_rerender = true;
    }

    /// Scalar field for the active magnetic view, normalized to [0, 1], plus
    /// its colormap. Mirrors `graphene_view_scalar` / `cooper_view_scalar`.
    #[allow(clippy::type_complexity)]
    fn magnetic_view_scalar(&self) -> Option<(Vec<f64>, fn(f64) -> [u8; 4])> {
        let d = self.density_result.as_ref()?;
        let vr = self.vortex_result.as_ref()?;
        let combined =
            moire_core::magnetic::combined_gap_with_vortices(&d.gap_field, &vr.suppression_field);
        match self.magnetic_view {
            MagneticView::CombinedGap => {
                let norm = normalize_to_unit_range(&combined)
                    .unwrap_or_else(|| vec![0.5; combined.len()]);
                Some((norm, self.cmap(moire_core::colormap::coolwarm)))
            }
            MagneticView::Susceptibility => {
                // χ ∈ [-1, 0]; shift to [0, 1] for the plasma map.
                let chi = moire_core::magnetic::local_susceptibility(&combined);
                let data = chi.iter().map(|&c| (c + 1.0).clamp(0.0, 1.0)).collect();
                Some((data, self.cmap(moire_core::colormap::plasma)))
            }
            MagneticView::ScreeningCurrent => {
                // Already peak-normalized to [0, 1] in recompute_magnetic.
                let field = self.screening_field.as_ref()?;
                Some((field.clone(), self.cmap(moire_core::colormap::plasma)))
            }
        }
    }

    /// Clear all Cooper-derived caches and textures (used when inputs are
    /// missing or the proximity decay errors out).
    fn clear_cooper_state(&mut self) {
        self.cooper_gap = None;
        self.cooper_z_coords = None;
        self.cooper_decay = None;
        self.cooper_texture = None;
        self.majorana_density = None;
    }

    /// Build the Cooper 3D / proximity state from the current density + vortex
    /// suppression. The full nz*N*N volume is never materialized: the field is
    /// separable (`gap_2d[ixy] * decay[iz]`), so any z-slice is one multiply
    /// pass and the global value range comes from the gap/decay extremes. The
    /// speculative Majorana volume is the exception — it is cached lazily and
    /// only when its view or toggle is active.
    fn recompute_cooper(&mut self, ctx: &egui::Context) {
        // Combine gap with vortex suppression while only borrowing self
        // immutably; the result is owned so the borrows end before we mutate.
        let prepared = self
            .density_result
            .as_ref()
            .zip(self.vortex_result.as_ref())
            .map(|(density, vortex)| {
                let combined = moire_core::magnetic::combined_gap_with_vortices(
                    &density.gap_field,
                    &vortex.suppression_field,
                );
                (combined, density.resolution, vortex.vortex_positions.clone())
            });
        let Some((combined, resolution, vortex_positions)) = prepared else {
            self.clear_cooper_state();
            return;
        };

        let extent = self
            .moire_result
            .as_ref()
            .map(|m| m.physical_extent)
            .unwrap_or(self.physical_extent);

        let z_coords = moire_core::topological::z_grid(&self.proximity_config);
        let decay = match moire_core::topological::proximity_decay_profile(
            &z_coords,
            self.proximity_config.xi_prox,
            self.proximity_config.interface_transparency,
        ) {
            Ok(d) => d,
            Err(e) => {
                // Slider ranges keep the inputs valid, so this is defensive.
                eprintln!("Cooper proximity decay error: {e}");
                self.clear_cooper_state();
                return;
            }
        };

        let n_z = z_coords.len().max(1);
        self.z_slice_index = self.z_slice_index.min(n_z - 1);

        // Majorana is speculative and expensive: compute the full 3D volume
        // once (cached until the next magnetic recompute) and only when its
        // view or toggle asks for it. An empty vortex list yields all zeros.
        if (self.show_majorana || self.cooper_view == CooperView::Majorana)
            && self.majorana_density.is_none()
        {
            match moire_core::topological::majorana_probability_density_3d(
                resolution,
                extent,
                &z_coords,
                &vortex_positions,
                DEFAULT_MAJORANA_XI,
                DEFAULT_KF,
                self.proximity_config.xi_prox,
            ) {
                Ok(vol) => self.majorana_density = Some(vol),
                Err(e) => eprintln!("Cooper Majorana error: {e}"),
            }
        }

        self.cooper_gap = Some(combined);
        self.cooper_z_coords = Some(z_coords);
        self.cooper_decay = Some(decay);

        if let Some((data, cmap)) = self.cooper_view_scalar() {
            self.cooper_texture = Some(render::pattern::create_texture(
                ctx,
                "cooper_view",
                &data,
                resolution,
                cmap,
            ));
        } else {
            // DecayProfile renders as a line plot, not a texture.
            self.cooper_texture = None;
        }
        self.needs_surface_rerender = true;
    }

    /// Scalar field for the active Cooper view, normalized to [0, 1], plus its
    /// colormap. Gap views (`InterfaceGap`, `ZSlice`) normalize against the
    /// global 3D range so slices dim with depth; `Majorana` returns the cached
    /// volume's z-slice (already [0, 1]); `DecayProfile` returns `None`.
    #[allow(clippy::type_complexity)]
    fn cooper_view_scalar(&self) -> Option<(Vec<f64>, fn(f64) -> [u8; 4])> {
        let z_coords = self.cooper_z_coords.as_ref()?;
        match self.cooper_view {
            CooperView::InterfaceGap | CooperView::ZSlice => {
                let combined = self.cooper_gap.as_ref()?;
                let decay = self.cooper_decay.as_ref()?;
                let iz = match self.cooper_view {
                    CooperView::ZSlice => self.z_slice_index.min(z_coords.len().saturating_sub(1)),
                    // Interface layer: first z >= 0.
                    _ => z_coords.iter().position(|&z| z >= 0.0).unwrap_or(0),
                };
                let f = decay.get(iz).copied().unwrap_or(0.0);
                let (lo, hi) = cooper_global_range(combined, decay);
                let range = if (hi - lo).abs() < 1e-15 { 1.0 } else { hi - lo };
                let data = combined
                    .iter()
                    .map(|&g| (((g * f) - lo) / range).clamp(0.0, 1.0))
                    .collect();
                Some((data, self.cmap(moire_core::colormap::coolwarm)))
            }
            CooperView::Majorana => {
                let vol = self.majorana_density.as_ref()?;
                let n2 = self.cooper_gap.as_ref()?.len();
                let iz = self.z_slice_index.min(z_coords.len().saturating_sub(1));
                let start = iz * n2;
                let slice = vol.get(start..start + n2)?;
                Some((slice.to_vec(), self.cmap(moire_core::colormap::viridis)))
            }
            CooperView::DecayProfile => None,
        }
    }

    /// Global Delta(z) value range (meV) across the full 3D proximity volume,
    /// used to normalize z-slices and label the colorbar. `None` until the
    /// Cooper stage has run.
    pub fn cooper_value_range(&self) -> Option<(f64, f64)> {
        let combined = self.cooper_gap.as_ref()?;
        let decay = self.cooper_decay.as_ref()?;
        Some(cooper_global_range(combined, decay))
    }

    /// Run the curvature + graphene stack (or supermoire) + BM computation
    /// and rebuild the active-view texture. Curvature runs first because its
    /// height field feeds the optional displacement warp of the stack.
    fn recompute_graphene(&mut self, ctx: &egui::Context) {
        // Persisted state may predate the valley field; only +/-1 is valid.
        if self.graphene_valley != -1 {
            self.graphene_valley = 1;
        }

        let mut ccfg = self.curvature_config;
        ccfg.resolution = self.resolution;
        ccfg.physical_extent = self.physical_extent;
        ccfg.valley = self.graphene_valley;
        let curvature = match moire_core::curvature::compute_curvature(&ccfg) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Curvature computation error: {e}");
                return;
            }
        };

        let displacement = if self.graphene_warp && ccfg.geometry != CurvatureGeometry::Flat {
            let dx = self.physical_extent / (self.resolution - 1).max(1) as f64;
            Some(moire_core::curvature::displacement_field(
                &curvature.height,
                self.resolution,
                dx,
            ))
        } else {
            None
        };

        let stack_cfg = GrapheneStackConfigV2 {
            base: GrapheneStackConfig {
                stacking: self.graphene_stack,
                twist_angle_deg: self.graphene_twist,
                lattice_a: moire_core::graphene::GRAPHENE_A,
                resolution: self.resolution,
                physical_extent: self.physical_extent,
            },
            honeycomb: self.graphene_honeycomb,
            heterostrain_percent: self.graphene_strain_percent,
            heterostrain_angle_deg: self.graphene_strain_angle,
        };

        let stack = if let Some(mat) = self.supermoire_material() {
            // compute_supermoire has no displacement input, so the curvature
            // warp applies only to the plain stack path.
            let sm_cfg = SupermoireConfig {
                stack: stack_cfg,
                overlayer_a: mat.a,
                overlayer_lattice_type: mat.lattice_type,
                interface_twist_deg: self.supermoire_interface_twist,
            };
            let sm = match moire_core::graphene::compute_supermoire(&sm_cfg) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Supermoire computation error: {e}");
                    return;
                }
            };
            let stack = GrapheneStackResult {
                pattern: sm.pattern.clone(),
                resolution: sm.resolution,
                physical_extent: sm.physical_extent,
                moire_period: sm.stack_period,
                n_layers: sm.n_layers,
            };
            self.supermoire_result = Some(sm);
            stack
        } else {
            self.supermoire_result = None;
            let disp = displacement
                .as_ref()
                .map(|(u_x, u_y)| (u_x.as_slice(), u_y.as_slice()));
            match moire_core::graphene::compute_graphene_stack_v2(&stack_cfg, disp) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Graphene stack computation error: {e}");
                    return;
                }
            }
        };

        let delta_max = if self.graphene_stack.is_twisted() && self.graphene_twist > 0.0 {
            let fb_cfg = FlatBandConfig {
                twist_angle_deg: self.graphene_twist,
                n_layers: self.graphene_stack.n_layers(),
                filling: self.graphene_filling,
            };
            match moire_core::graphene::compute_flat_band_sc(&fb_cfg) {
                Ok(fb) => fb.delta_mev,
                Err(e) => {
                    eprintln!("Flat-band computation error: {e}");
                    0.0
                }
            }
        } else {
            0.0
        };

        self.graphene_fft =
            match moire_core::fft::compute_fft_2d(&stack.pattern, stack.resolution) {
                Ok(f) => Some(f),
                Err(e) => {
                    eprintln!("Graphene FFT computation error: {e}");
                    None
                }
            };

        self.recompute_bm_model();

        // Gap modulation mirrors the Python gap_modulation convention with
        // amplitude delta_max / 2, then curvature suppression applied.
        use std::f64::consts::PI;
        let gap: Vec<f64> = stack
            .pattern
            .iter()
            .zip(curvature.gap_suppression.iter())
            .map(|(&p, &s)| (delta_max + 0.5 * delta_max * (PI + PI * p).cos()) * s)
            .collect();

        self.graphene_result = Some(stack);
        self.curvature_result = Some(curvature);
        self.graphene_gap = Some(gap);

        if let Some((data, cmap)) = self.graphene_view_scalar() {
            self.graphene_texture = Some(render::pattern::create_texture(
                ctx,
                "graphene_view",
                &data,
                self.resolution,
                cmap,
            ));
        }
        self.needs_surface_rerender = true;
    }

    /// Lazily compute the Bistritzer-MacDonald band structure / DOS for the
    /// Bands and Dos views (each costs ~1 s, so untouched views are skipped)
    /// and invalidate the caches whenever the effective twist or valley
    /// changes. Untwisted stacks fall back to the magic angle of the same
    /// layer count.
    fn recompute_bm_model(&mut self) {
        let effective_twist = if self.graphene_stack.is_twisted() && self.graphene_twist > 0.0 {
            self.graphene_twist
        } else {
            moire_core::graphene::magic_angle_deg(self.graphene_stack.n_layers())
                .unwrap_or(default_graphene_twist())
        };

        let bm_key = (effective_twist, self.graphene_valley);
        if self.bm_cache_key != Some(bm_key) {
            self.band_structure = None;
            self.dos_result = None;
            self.bm_cache_key = Some(bm_key);
        }

        let bm_cfg = BMConfig {
            twist_angle_deg: effective_twist,
            valley: self.graphene_valley,
            ..Default::default()
        };

        if self.graphene_view == GrapheneView::Bands && self.band_structure.is_none() {
            match moire_core::bm_model::compute_band_structure(&bm_cfg, BM_N_K_PER_SEGMENT) {
                Ok(bs) => self.band_structure = Some(bs),
                Err(e) => eprintln!("Band structure computation error: {e}"),
            }
        }
        if self.graphene_view == GrapheneView::Dos && self.dos_result.is_none() {
            match moire_core::bm_model::compute_dos(
                &bm_cfg,
                BM_N_K_GRID,
                DOS_E_WINDOW_MEV,
                DOS_N_BINS,
                DOS_BROADENING_MEV,
            ) {
                Ok(d) => self.dos_result = Some(d),
                Err(e) => eprintln!("DOS computation error: {e}"),
            }
        }
    }

    /// Scalar field for the active graphene view, normalized to [0, 1], plus
    /// its colormap. The pseudo-field is mapped symmetrically about 0.5 so
    /// zero field stays at the colormap midpoint.
    #[allow(clippy::type_complexity)]
    fn graphene_view_scalar(&self) -> Option<(Vec<f64>, fn(f64) -> [u8; 4])> {
        match self.graphene_view {
            GrapheneView::Pattern => {
                let g = self.graphene_result.as_ref()?;
                Some((g.pattern.clone(), self.cmap(moire_core::colormap::viridis)))
            }
            GrapheneView::GapMap => {
                let gap = self.graphene_gap.as_ref()?;
                let norm = normalize_to_unit_range(gap)
                    .unwrap_or_else(|| vec![0.5; gap.len()]);
                Some((norm, self.cmap(moire_core::colormap::coolwarm)))
            }
            GrapheneView::PseudoField => {
                let c = self.curvature_result.as_ref()?;
                let data = if c.max_abs_field < 1e-15 {
                    vec![0.5; c.pseudo_field.len()]
                } else {
                    c.pseudo_field
                        .iter()
                        .map(|&b| 0.5 + 0.5 * b / c.max_abs_field)
                        .collect()
                };
                Some((data, self.cmap(moire_core::colormap::coolwarm)))
            }
            GrapheneView::Strain => {
                let c = self.curvature_result.as_ref()?;
                let mag: Vec<f64> = c
                    .strain_xx
                    .iter()
                    .zip(&c.strain_yy)
                    .zip(&c.strain_xy)
                    .map(|((&xx, &yy), &xy)| (xx * xx + yy * yy + 2.0 * xy * xy).sqrt())
                    .collect();
                let norm = normalize_to_unit_range(&mag).unwrap_or_else(|| vec![0.0; mag.len()]);
                Some((norm, self.cmap(moire_core::colormap::viridis)))
            }
            // Already log-scaled to [0, 1]; same colormap as the Fourier tab.
            GrapheneView::Fourier => {
                let f = self.graphene_fft.as_ref()?;
                Some((f.clone(), self.cmap(moire_core::colormap::inferno)))
            }
            // Bands and DOS render as egui_plot lines, not textures, so the
            // texture / 3D-surface / screenshot paths all skip gracefully.
            GrapheneView::Bands | GrapheneView::Dos => None,
        }
    }

    /// Compute and render comparison textures for all 3 overlayers.
    fn refresh_comparison(&mut self, ctx: &egui::Context) {
        let surface_bg = if self.dark_mode {
            egui::Color32::from_rgb(30, 30, 35)
        } else {
            egui::Color32::from_rgb(240, 240, 245)
        };
        let substrate = self.substrate_material();
        let overlayers = materials::overlayers();
        let mut textures = Vec::with_capacity(6);
        let comp_res = 128_usize;
        let moire_cmap = self.cmap(moire_core::colormap::viridis);
        let density_cmap = self.cmap(moire_core::colormap::coolwarm);

        for mat in overlayers {
            let config = MoireConfig {
                substrate_a: substrate.a,
                substrate_lattice_type: substrate.lattice_type,
                overlayer_a: mat.a,
                overlayer_lattice_type: mat.lattice_type,
                twist_angle_deg: self.twist_angle,
                resolution: comp_res,
                physical_extent: self.physical_extent,
                dw_factor_substrate: 1.0,
                dw_factor_overlayer: 1.0,
            };
            let moire = match moire_core::moire::compute_moire(&config) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("Comparison moire computation error: {e}");
                    continue;
                }
            };

            if self.view_mode == ViewMode::Surface3D {
                let comp_opts = self.surface_opts();
                let img = render::surface3d::render_surface_3d_opts(
                    &moire.pattern,
                    comp_res,
                    256,
                    256,
                    &self.camera,
                    moire_cmap,
                    surface_bg,
                    &comp_opts,
                );
                textures.push(ctx.load_texture("comp_moire", img, egui::TextureOptions::LINEAR));
            } else {
                textures.push(render::pattern::create_texture(
                    ctx,
                    "comp_moire",
                    &moire.pattern,
                    comp_res,
                    moire_cmap,
                ));
            }

            let density =
                match moire_core::density::compute_density_modulation(&moire, &self.density_config)
                {
                    Ok(d) => d,
                    Err(e) => {
                        eprintln!("Comparison density computation error: {e}");
                        continue;
                    }
                };
            let d_norm: Vec<f64> = normalize_to_unit_range(&density.gap_field)
                .unwrap_or_else(|| vec![0.5; density.gap_field.len()]);

            if self.view_mode == ViewMode::Surface3D {
                let comp_opts = self.surface_opts();
                let img = render::surface3d::render_surface_3d_opts(
                    &d_norm,
                    comp_res,
                    256,
                    256,
                    &self.camera,
                    density_cmap,
                    surface_bg,
                    &comp_opts,
                );
                textures.push(ctx.load_texture("comp_density", img, egui::TextureOptions::LINEAR));
            } else {
                textures.push(render::pattern::create_texture(
                    ctx,
                    "comp_density",
                    &d_norm,
                    comp_res,
                    density_cmap,
                ));
            }
        }

        self.comparison_textures = Some(textures);
    }
}

impl eframe::App for MoireApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keyboard shortcuts + menu bar. Both surface the same MenuAction
        // enum so the routing below handles either equivalently. Done before
        // the compute pass so Reset/Wireframe toggles take effect this frame.
        let kbd_action = ui::menu::detect_shortcut(ctx);
        let menu_action = ui::menu::show(ctx, self);
        if let Some(act) = menu_action.or(kbd_action) {
            self.apply_menu_action(act, ctx);
        }

        // Z-sweep animation: advance the Cooper z-slice ~3 Hz while playing.
        // The desktop has no isosurface, so the honest analogue of the web's
        // iso-range sweep is stepping the z-slice (one multiply pass + texture).
        if self.cooper_playing
            && self.active_tab == Tab::CooperSurface3D
            && self.cooper_view == CooperView::ZSlice
        {
            let n_z = self
                .cooper_z_coords
                .as_ref()
                .map(|z| z.len())
                .unwrap_or(self.proximity_config.n_z_layers)
                .max(1);
            let now = ctx.input(|i| i.time);
            if now - self.cooper_last_tick > 0.35 {
                self.z_slice_index = (self.z_slice_index + 1) % n_z;
                self.cooper_last_tick = now;
                self.needs_cooper_recompute = true;
            }
            ctx.request_repaint_after(std::time::Duration::from_millis(350));
        }

        if self.needs_recompute {
            self.recompute(ctx);
            self.needs_magnetic_recompute = true;
            self.needs_graphene_recompute = true;
            self.needs_recompute = false;
        }

        if self.needs_magnetic_recompute {
            self.recompute_magnetic(ctx);
            self.needs_magnetic_recompute = false;
        }

        if self.needs_graphene_recompute {
            self.recompute_graphene(ctx);
            self.needs_graphene_recompute = false;
        }

        // Cooper depends on the vortex suppression, so it runs after magnetic
        // (which sets `needs_cooper_recompute`).
        if self.needs_cooper_recompute {
            self.recompute_cooper(ctx);
            self.needs_cooper_recompute = false;
        }

        if self.needs_surface_rerender && self.view_mode == ViewMode::Surface3D {
            self.rerender_surface(ctx);
            self.needs_surface_rerender = false;
        }

        egui::SidePanel::left("controls")
            .min_width(240.0)
            .max_width(350.0)
            .show(ctx, |panel_ui| {
                egui::ScrollArea::vertical().show(panel_ui, |ui| {
                    ui::sidebar::show_sidebar(ui, self);
                    ui.separator();
                    ui::info_panel::show_info_panel(ui, self);
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui::viewport::show_viewport(ui, self);
        });

        if self.show_about {
            let mut open = self.show_about;
            ui::about::show(ctx, &mut open);
            self.show_about = open;
        }

        // Comparison window
        if self.show_comparison {
            if self.comparison_needs_refresh || self.comparison_textures.is_none() {
                self.refresh_comparison(ctx);
                self.comparison_needs_refresh = false;
            }

            let mut open = self.show_comparison;
            egui::Window::new("Substrate Comparison")
                .open(&mut open)
                .default_width(800.0)
                .default_height(500.0)
                .show(ctx, |ui| {
                    let overlayers = materials::overlayers();
                    let substrate = self.substrate_material();

                    if let Some(ref textures) = self.comparison_textures {
                        ui.horizontal(|ui| {
                            for (i, mat) in overlayers.iter().enumerate() {
                                ui.vertical(|ui| {
                                    ui.heading(format!("{} / {}", mat.formula, substrate.formula));
                                    ui.label(format!(
                                        "Mismatch: {:.1}%",
                                        (mat.a - substrate.a).abs() / substrate.a * 100.0
                                    ));
                                    let period =
                                        moire_core::moire::moire_periodicity_1d(mat.a, substrate.a);
                                    ui.label(format!("Period: {:.1} A", period));
                                    ui.add_space(4.0);

                                    let moire_idx = i * 2;
                                    let density_idx = i * 2 + 1;
                                    let tex_size = egui::vec2(220.0, 220.0);

                                    if moire_idx < textures.len() {
                                        ui.label("Moire Pattern:");
                                        ui.image(egui::load::SizedTexture::new(
                                            textures[moire_idx].id(),
                                            tex_size,
                                        ));
                                    }
                                    if density_idx < textures.len() {
                                        ui.label("Density:");
                                        ui.image(egui::load::SizedTexture::new(
                                            textures[density_idx].id(),
                                            tex_size,
                                        ));
                                    }
                                });
                                if i < overlayers.len() - 1 {
                                    ui.separator();
                                }
                            }
                        });
                    } else {
                        ui.label("Computing...");
                    }
                });
            self.show_comparison = open;
        }

        // Phase-diagram window (mirrors the comparison-window block; the module
        // owns its refresh + open-state handling).
        if self.show_phase_diagram {
            ui::phase_window::show(ctx, self);
        }
    }
}

#[cfg(test)]
// Test setup favors readable field-by-field mutation over struct-update syntax,
// especially for nested config fields (e.g. `app.magnetic_config.bz`).
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    /// Build a default app at a small resolution and run the full 2D compute
    /// pipeline against a headless `egui::Context`. `egui::Context::default()`
    /// allocates textures into the CPU-side texture manager, so no display or
    /// GPU is required (see `bin/capture.rs` for the headless precedent).
    /// Resolution 64 keeps each test well under a second.
    fn pipeline_app(resolution: usize) -> (MoireApp, egui::Context) {
        let mut app = MoireApp::default();
        app.resolution = resolution;
        let ctx = egui::Context::default();
        app.recompute(&ctx);
        app.recompute_magnetic(&ctx);
        app.recompute_graphene(&ctx);
        app.recompute_cooper(&ctx);
        (app, ctx)
    }

    #[test]
    fn test_recompute_populates_core_results() {
        let (app, _ctx) = pipeline_app(64);
        let n2 = 64 * 64;

        let moire = app.moire_result.as_ref().expect("moire result");
        assert_eq!(moire.pattern.len(), n2);
        let density = app.density_result.as_ref().expect("density result");
        assert_eq!(density.gap_field.len(), n2);
        let fft = app.fft_data.as_ref().expect("fft data");
        assert_eq!(fft.len(), n2);

        assert!(app.pattern_texture.is_some());
        assert!(app.density_texture.is_some());
        assert!(app.fft_texture.is_some());
    }

    #[test]
    fn test_recompute_magnetic_with_field() {
        let mut app = MoireApp::default();
        app.resolution = 64;
        app.magnetic_config.bz = 5.0;
        let ctx = egui::Context::default();
        app.recompute(&ctx);
        app.recompute_magnetic(&ctx);

        let vr = app.vortex_result.as_ref().expect("vortex result");
        assert!(
            !vr.vortex_positions.is_empty(),
            "Bz = 5 T should nucleate vortices"
        );
        assert_eq!(vr.suppression_field.len(), 64 * 64);
        assert!(app.zeeman_result.is_some());
        assert!(app.magnetic_texture.is_some());
    }

    #[test]
    fn test_recompute_graphene_populates() {
        let (app, _ctx) = pipeline_app(64);
        assert!(app.graphene_result.is_some());
        assert!(app.curvature_result.is_some());
        assert!(app.graphene_gap.is_some());
        assert!(app.graphene_texture.is_some());
    }

    #[test]
    fn test_persisted_state_backcompat() {
        // Simulate loading state saved before Track 2: serialize a default app,
        // drop the fields Track 2 added, and confirm it still deserializes with
        // those fields at their serde defaults. Guards the `#[serde(default)]`
        // discipline for every persisted field this work added.
        let app = MoireApp::default();
        let mut value = serde_json::to_value(&app).expect("serialize");
        let obj = value.as_object_mut().expect("object");
        for key in [
            "cooper_view",
            "colormap_override",
            "magnetic_view",
            "phase_b_max",
            "phase_delta_max",
            "phase_mu",
        ] {
            assert!(obj.remove(key).is_some(), "{key} was not serialized");
        }
        let restored: MoireApp = serde_json::from_value(value).expect("deserialize old blob");
        assert_eq!(restored.cooper_view, CooperView::default());
        assert_eq!(restored.colormap_override, None);
        assert_eq!(restored.magnetic_view, MagneticView::default());
        assert_eq!(restored.phase_b_max, 100.0);
        assert_eq!(restored.phase_delta_max, 10.0);
        assert_eq!(restored.phase_mu, 0.0);
    }

    #[test]
    fn test_capture_current_view_every_tab() {
        let (mut app, _ctx) = pipeline_app(64);
        let all_tabs = [
            Tab::Pattern,
            Tab::Density,
            Tab::Fourier,
            Tab::MagneticField,
            Tab::CooperSurface3D,
            Tab::Graphene,
        ];
        // With the default (non-plot) graphene/cooper views, every tab in both
        // view modes produces a capturable image.
        for &tab in &all_tabs {
            app.active_tab = tab;
            for vm in [ViewMode::Flat2D, ViewMode::Surface3D] {
                app.view_mode = vm;
                assert!(
                    app.capture_current_view().is_some(),
                    "{tab:?}/{vm:?} should capture"
                );
            }
        }

        // Plot-only views (Bands/Dos/DecayProfile) have no capturable texture.
        app.active_tab = Tab::Graphene;
        for gv in [GrapheneView::Bands, GrapheneView::Dos] {
            app.graphene_view = gv;
            for vm in [ViewMode::Flat2D, ViewMode::Surface3D] {
                app.view_mode = vm;
                assert!(
                    app.capture_current_view().is_none(),
                    "Graphene {gv:?}/{vm:?} should not capture"
                );
            }
        }
        app.graphene_view = GrapheneView::Pattern;

        app.active_tab = Tab::CooperSurface3D;
        app.cooper_view = CooperView::DecayProfile;
        for vm in [ViewMode::Flat2D, ViewMode::Surface3D] {
            app.view_mode = vm;
            assert!(
                app.capture_current_view().is_none(),
                "Cooper DecayProfile/{vm:?} should not capture"
            );
        }
    }

    #[test]
    fn test_rerender_surface_all_tabs() {
        let (mut app, ctx) = pipeline_app(64);
        app.view_mode = ViewMode::Surface3D;
        for tab in [
            Tab::Pattern,
            Tab::Density,
            Tab::Fourier,
            Tab::MagneticField,
            Tab::CooperSurface3D,
            Tab::Graphene,
        ] {
            app.active_tab = tab;
            app.surface_texture = None;
            app.rerender_surface(&ctx);
            assert!(
                app.surface_texture.is_some(),
                "no surface texture rendered for {tab:?}"
            );
        }
    }

    #[test]
    fn test_recompute_cooper_populates() {
        let (app, _ctx) = pipeline_app(64);
        assert!(app.cooper_gap.is_some());
        assert!(app.cooper_z_coords.is_some());
        assert!(app.cooper_decay.is_some());
        // Default view is InterfaceGap, which builds a texture.
        assert!(app.cooper_texture.is_some());
        let n_z = app.cooper_z_coords.as_ref().unwrap().len();
        assert_eq!(n_z, app.proximity_config.n_z_layers);
        assert_eq!(app.cooper_decay.as_ref().unwrap().len(), n_z);
        assert_eq!(app.cooper_gap.as_ref().unwrap().len(), 64 * 64);
    }

    #[test]
    fn test_cooper_slice_equals_gap_times_decay() {
        // Cross-validate the lightweight separable slice path against the core
        // volume path `compute_gap_3d`, term by term.
        let (app, _ctx) = pipeline_app(32);
        let combined = app.cooper_gap.as_ref().unwrap();
        let decay = app.cooper_decay.as_ref().unwrap();
        let vol =
            moire_core::topological::compute_gap_3d(combined, 32, &app.proximity_config).unwrap();
        let n2 = 32 * 32;
        for iz in [0usize, decay.len() / 2, decay.len() - 1] {
            let f = decay[iz];
            for ixy in (0..n2).step_by(97) {
                let lightweight = combined[ixy] * f;
                let volume = vol.gap_3d[iz * n2 + ixy];
                assert!(
                    (lightweight - volume).abs() < 1e-12,
                    "mismatch iz={iz} ixy={ixy}: {lightweight} vs {volume}"
                );
            }
        }
    }

    #[test]
    fn test_cooper_view_scalar_all_views() {
        let mut app = MoireApp::default();
        app.resolution = 32;
        app.magnetic_config.bz = 5.0; // nucleate vortices for a non-trivial Majorana
        app.show_majorana = true; // populate the Majorana cache
        let ctx = egui::Context::default();
        app.recompute(&ctx);
        app.recompute_magnetic(&ctx);
        app.recompute_cooper(&ctx);

        let n2 = 32 * 32;
        for view in [
            CooperView::InterfaceGap,
            CooperView::ZSlice,
            CooperView::Majorana,
        ] {
            app.cooper_view = view;
            let (data, _cmap) = app.cooper_view_scalar().expect("scalar for view");
            assert_eq!(data.len(), n2, "wrong length for {view:?}");
            for &v in &data {
                assert!((0.0..=1.0).contains(&v), "value {v} out of [0,1] for {view:?}");
            }
        }

        app.cooper_view = CooperView::DecayProfile;
        assert!(app.cooper_view_scalar().is_none());
    }

    #[test]
    fn test_majorana_gated() {
        let mut app = MoireApp::default();
        app.resolution = 32;
        app.magnetic_config.bz = 5.0;
        let ctx = egui::Context::default();
        app.recompute(&ctx);
        app.recompute_magnetic(&ctx);

        // Default view = InterfaceGap, show_majorana = false → no volume.
        app.recompute_cooper(&ctx);
        assert!(
            app.majorana_density.is_none(),
            "Majorana computed while gated off"
        );

        // Enabling the toggle populates the cache on the next Cooper pass.
        app.show_majorana = true;
        app.needs_cooper_recompute = true;
        app.recompute_cooper(&ctx);
        assert!(
            app.majorana_density.is_some(),
            "Majorana not computed when enabled"
        );
    }

    #[test]
    fn test_fft_peaks_populated_and_sorted() {
        let (app, _ctx) = pipeline_app(128);
        let peaks = app.fft_peaks.as_ref().expect("fft peaks");
        assert!(!peaks.is_empty(), "expected at least one FFT peak");
        assert!(peaks.len() <= FFT_MAX_PEAKS);
        for w in peaks.windows(2) {
            assert!(
                w[0].amplitude >= w[1].amplitude,
                "peaks not amplitude-descending"
            );
        }
    }

    #[test]
    fn test_fft_peak_wavelength_matches_lattice() {
        // Physics/wiring anchor: the FFT of the product pattern is dominated by
        // the constituent reciprocal-lattice vectors, so some detected peak's
        // wavelength (2π/|k|, using `fft_frequencies`) lands within ~20% of a
        // constituent lattice constant. A wrong frequency scale would break it.
        //
        // (The moire beat period is NOT the strongest peak for this pattern —
        // the atomic-lattice harmonics dominate — so we anchor on the lattice
        // constants, not the moire period.)
        let (app, _ctx) = pipeline_app(128);
        let peaks = app.fft_peaks.as_ref().expect("fft peaks");
        let a_sub = app.substrate_material().a;
        let a_over = app.overlayer_material().a;
        let matched = peaks.iter().any(|p| {
            let k_mag = (p.kx * p.kx + p.ky * p.ky).sqrt();
            if k_mag <= 1e-9 {
                return false;
            }
            let lambda = 2.0 * std::f64::consts::PI / k_mag;
            [a_sub, a_over]
                .iter()
                .any(|&a| (lambda - a).abs() / a < 0.2)
        });
        assert!(
            matched,
            "no FFT peak within 20% of a_sub={a_sub:.3} or a_over={a_over:.3}"
        );
    }

    #[test]
    fn test_phase_refresh_builds_texture() {
        let mut app = MoireApp::default();
        let ctx = egui::Context::default();
        app.phase_b_max = 100.0;
        app.phase_delta_max = 10.0;
        app.phase_mu = 0.0;
        crate::ui::phase_window::refresh_phase(&ctx, &mut app);
        let tex = app.phase_texture.as_ref().expect("phase texture");
        assert_eq!(tex.size(), [160, 160]);
    }

    #[test]
    fn test_magnetic_view_scalar_all_views() {
        let mut app = MoireApp::default();
        app.resolution = 32;
        app.magnetic_config.bz = 5.0;
        let ctx = egui::Context::default();
        app.recompute(&ctx);
        let n2 = 32 * 32;
        for view in [
            MagneticView::CombinedGap,
            MagneticView::Susceptibility,
            MagneticView::ScreeningCurrent,
        ] {
            app.magnetic_view = view;
            app.recompute_magnetic(&ctx);
            let (data, _cmap) = app.magnetic_view_scalar().expect("scalar for view");
            assert_eq!(data.len(), n2, "wrong length for {view:?}");
            for &v in &data {
                assert!((0.0..=1.0).contains(&v), "value {v} out of [0,1] for {view:?}");
            }
        }
    }

    #[test]
    fn test_screening_populated_when_selected() {
        let mut app = MoireApp::default();
        app.resolution = 32;
        app.magnetic_config.bz = 5.0;
        let ctx = egui::Context::default();
        app.recompute(&ctx);

        // Not selected → the expensive screening field stays uncomputed.
        app.magnetic_view = MagneticView::CombinedGap;
        app.recompute_magnetic(&ctx);
        assert!(app.screening_field.is_none());

        // Selected → cache populated, peak-normalized to [0, 1].
        app.magnetic_view = MagneticView::ScreeningCurrent;
        app.recompute_magnetic(&ctx);
        let sf = app.screening_field.as_ref().expect("screening field");
        assert_eq!(sf.len(), 32 * 32);
        let max = sf.iter().copied().fold(0.0_f64, f64::max);
        assert!(max > 0.0 && max <= 1.0 + 1e-9, "screening max = {max}");
    }

    #[test]
    fn test_zeeman_uses_app_g_factor() {
        let mut app = MoireApp::default();
        app.resolution = 32;
        app.magnetic_config.bx = 1.0; // in-plane field → nonzero Zeeman
        let ctx = egui::Context::default();
        app.recompute(&ctx);

        app.g_factor = 10.0;
        app.recompute_magnetic(&ctx);
        let e1 = app.zeeman_result.as_ref().unwrap().zeeman_energy;
        app.g_factor = 20.0;
        app.recompute_magnetic(&ctx);
        let e2 = app.zeeman_result.as_ref().unwrap().zeeman_energy;

        assert!(e1 > 0.0, "expected nonzero Zeeman, got {e1}");
        assert!((e2 - 2.0 * e1).abs() < 1e-9, "E_Z not linear in g: e1={e1}, e2={e2}");
    }

    #[test]
    fn test_cmap_resolver_auto_and_override() {
        let mut app = MoireApp::default();
        let viridis = moire_core::colormap::viridis as fn(f64) -> [u8; 4];
        let plasma = moire_core::colormap::plasma as fn(f64) -> [u8; 4];

        // Auto: the semantic palette passes through unchanged.
        app.colormap_override = None;
        assert_eq!(app.cmap(viridis) as usize, viridis as usize);

        // Override: every semantic maps to the chosen palette.
        app.colormap_override = Some(ColormapName::Plasma);
        for semantic in [
            viridis,
            moire_core::colormap::coolwarm as fn(f64) -> [u8; 4],
            moire_core::colormap::inferno as fn(f64) -> [u8; 4],
        ] {
            assert_eq!(app.cmap(semantic) as usize, plasma as usize);
        }
    }

    #[test]
    fn test_z_slice_clamped() {
        let mut app = MoireApp::default();
        app.resolution = 32;
        let ctx = egui::Context::default();
        app.recompute(&ctx);
        app.recompute_magnetic(&ctx);
        app.recompute_cooper(&ctx);

        // Push the index out of range, then shrink the z layering.
        app.z_slice_index = 9999;
        app.proximity_config.n_z_layers = 8;
        app.needs_cooper_recompute = true;
        app.recompute_cooper(&ctx);

        let new_n_z = app.cooper_z_coords.as_ref().unwrap().len();
        assert_eq!(new_n_z, 8);
        assert!(
            app.z_slice_index < new_n_z,
            "z_slice_index {} not clamped to {}",
            app.z_slice_index,
            new_n_z
        );
    }
}
