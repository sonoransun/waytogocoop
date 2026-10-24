//! `MoireApp` — the egui/eframe application state machine.
//!
//! Recompute-on-change: user mutations set `needs_recompute` / `needs_*`
//! flags, and the next `update()` tick runs the pipeline (isotope → moire →
//! density → FFT → textures → 3D re-render). The Spin Fluctuations stage, the
//! costliest, runs only while its tab is open and otherwise stays dirty. Menu
//! bar and keyboard shortcuts route through `MenuAction` in `ui::menu`; the 2D
//! viewport reads axis and colorbar metadata from `ui::viewport::view_meta`.

use std::ops::RangeInclusive;
use std::sync::Arc;

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
use moire_core::pdw::{
    PdwConfig, PdwFields, PdwGeometry, PdwPeak, PdwRegime, PdwState, PdwSweep, KAPPA_HARM_DEFAULT,
    KAPPA_H_DEFAULT, KAPPA_P_DEFAULT, KAPPA_Q_DEFAULT, TC_UTE2, T_PDW_UTE2,
};
use moire_core::spin_fluctuation::{
    self as sf, SfCoherence, SfConfig, SfDopingPoint, SfDopingRegime, SfFermiSurface, SfGlueConfig,
    SfGlueResult, SfInelasticForm, SfMagnetoresistance, SfPairingForm, SfPairingMap,
    SfResistivityFit,
};
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

/// Points per PDW temperature / field sweep (matches the web page).
const PDW_SWEEP_POINTS: usize = 241;
/// PDW regime-map grid per axis.
const PDW_REGIME_N: usize = 121;
/// Sweeps run to this multiple of the higher transition temperature.
const PDW_T_RANGE_FACTOR: f64 = 1.25;
/// Field sweeps run to this multiple of the larger critical field.
const PDW_B_RANGE_FACTOR: f64 = 1.15;

/// Points per SF temperature sweep over 2..=100 K: rho(0, T) at 1 K and R_H(T)
/// at 2 K (matches the web page).
const SF_RHO_T_POINTS: usize = 99;
const SF_HALL_T_POINTS: usize = 50;
/// Operating-point Tc* samples of the imprint curve, over the Tc-target
/// slider range.
const SF_IMPRINT_POINTS: usize = 80;
/// lambda samples per pairing-map curve on the coupling plot.
const SF_PAIRING_CURVE_POINTS: usize = 241;
/// Upper end of the coupling plot's lambda axis (lambda_tr(0.116) = 2.71).
const SF_PAIRING_LAMBDA_MAX: f64 = 3.0;

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

/// Map a signed field to [0, 1] symmetrically about 0.5, so zero stays at the
/// colormap midpoint.
fn symmetric_unit_range(data: &[f64]) -> Vec<f64> {
    let max_abs = data.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if max_abs < 1e-15 {
        return vec![0.5; data.len()];
    }
    data.iter().map(|&v| 0.5 + 0.5 * v / max_abs).collect()
}

/// Apply a separable Hann window (numpy `hanning` convention) to a row-major
/// NxN field before an STM-style FFT, suppressing leakage from incommensurate
/// peaks.
fn hann_taper(data: &[f64], n: usize) -> Vec<f64> {
    let w: Vec<f64> = (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1).max(1) as f64).cos())
        .collect();
    data.iter()
        .enumerate()
        .map(|(idx, &v)| v * w[idx / n] * w[idx % n])
        .collect()
}

fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![a];
    }
    (0..n)
        .map(|i| a + (b - a) * i as f64 / (n - 1) as f64)
        .collect()
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Tab {
    #[default]
    Pattern,
    Density,
    Fourier,
    MagneticField,
    CooperSurface3D,
    Graphene,
    SpinFluctuation,
    PairDensityWave,
}

/// 2D flat or 3D surface view mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum ViewMode {
    #[default]
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

/// Which field or plot the Pair Density Wave tab displays. The sweeps and the
/// regime map live on their own axes, not the moire grid, so they render as
/// plots (like the graphene Bands / Dos views).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PdwView {
    /// Composite charge density rho(r).
    #[default]
    ChargeDensity,
    /// Pair amplitude |d(r)| (meV).
    PairAmplitude,
    /// SPECULATIVE composite spin density S_z(r).
    SpinDensity,
    /// Hann-tapered FFT of rho(r), STM style.
    Fourier,
    /// Peak intensities vs temperature.
    TemperatureSweep,
    /// Peak intensities vs |B| along the sweep axis.
    FieldSweep,
    /// (|B|, T) regime map along the sweep axis.
    RegimeMap,
}

impl PdwView {
    pub const ALL: [PdwView; 7] = [
        PdwView::ChargeDensity,
        PdwView::PairAmplitude,
        PdwView::SpinDensity,
        PdwView::Fourier,
        PdwView::TemperatureSweep,
        PdwView::FieldSweep,
        PdwView::RegimeMap,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PdwView::ChargeDensity => "Charge density ρ(r)",
            PdwView::PairAmplitude => "Pair amplitude |d(r)|",
            PdwView::SpinDensity => "Spin density S_z(r) (spec.)",
            PdwView::Fourier => "STM-style FFT of ρ(r)",
            PdwView::TemperatureSweep => "Peaks vs temperature",
            PdwView::FieldSweep => "Peaks vs field",
            PdwView::RegimeMap => "Regime map (|B|, T)",
        }
    }

    /// Views drawn on their own axes rather than as moire-grid textures.
    pub fn is_plot(&self) -> bool {
        matches!(
            self,
            PdwView::TemperatureSweep | PdwView::FieldSweep | PdwView::RegimeMap
        )
    }
}

/// Field-sweep / regime-map direction in the paper's vector-magnet frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PdwSweepAxis {
    X,
    Y,
    #[default]
    Z,
    /// Along the applied field (z when the field is zero).
    AlongField,
}

impl PdwSweepAxis {
    pub const ALL: [PdwSweepAxis; 4] = [
        PdwSweepAxis::X,
        PdwSweepAxis::Y,
        PdwSweepAxis::Z,
        PdwSweepAxis::AlongField,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PdwSweepAxis::X => "Bx (≈ c* in UTe2)",
            PdwSweepAxis::Y => "By (= a in UTe2)",
            PdwSweepAxis::Z => "Bz (≈ b* in UTe2)",
            PdwSweepAxis::AlongField => "Along the applied B",
        }
    }

    /// How the axis reads in a title ("|B| along ..."): along a zero field
    /// the sweep falls back to z, and the title says so.
    pub fn phrase(&self, field: [f64; 3]) -> &'static str {
        match self {
            PdwSweepAxis::AlongField if field.iter().any(|&b| b != 0.0) => "the applied B",
            PdwSweepAxis::AlongField => PdwSweepAxis::Z.label(),
            axis => axis.label(),
        }
    }

    /// Unit sweep direction for the applied `field`.
    pub fn direction(&self, field: [f64; 3]) -> [f64; 3] {
        let norm = (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]).sqrt();
        match self {
            PdwSweepAxis::X => [1.0, 0.0, 0.0],
            PdwSweepAxis::Y => [0.0, 1.0, 0.0],
            PdwSweepAxis::AlongField if norm > 0.0 => field.map(|b| b / norm),
            PdwSweepAxis::Z | PdwSweepAxis::AlongField => [0.0, 0.0, 1.0],
        }
    }
}

/// User-tunable PDW settings; the material constants (Tc, critical fields,
/// wavevectors) come from the geometry, so switching geometry keeps these.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PdwSettings {
    pub geometry: PdwGeometry,
    /// Temperature in K.
    pub temperature: f64,
    /// T_PDW / Tc; 1 means no PDW-only window.
    pub tpdw_ratio: f64,
    /// Applied field (x ~ c*, y = a, z ~ b*) in Tesla.
    pub field: [f64; 3],
    pub sweep_axis: PdwSweepAxis,
    pub kappa_p: f64,
    pub kappa_q: f64,
    pub kappa_h: f64,
    pub kappa_harm: f64,
    /// log10 of the |rho|^2 detection floor.
    pub floor_log10: f64,
}

impl Default for PdwSettings {
    fn default() -> Self {
        Self {
            geometry: PdwGeometry::Ute2Surface,
            temperature: 0.3,
            tpdw_ratio: T_PDW_UTE2 / TC_UTE2,
            field: [0.0; 3],
            sweep_axis: PdwSweepAxis::Z,
            kappa_p: KAPPA_P_DEFAULT,
            kappa_q: KAPPA_Q_DEFAULT,
            kappa_h: KAPPA_H_DEFAULT,
            kappa_harm: KAPPA_HARM_DEFAULT,
            floor_log10: -2.0,
        }
    }
}

/// Cached PDW computation behind the Pair Density Wave tab.
pub struct PdwRuntime {
    pub config: PdwConfig,
    pub state: PdwState,
    pub peaks: Vec<PdwPeak>,
    pub fields: PdwFields,
    /// Log-normalized FFT of the tapered charge density; only computed while
    /// the Fourier view is active.
    pub fft: Option<Vec<f64>>,
    /// Unit field direction of the field sweep and regime map.
    pub direction: [f64; 3],
    pub t_sweep: PdwSweep,
    pub b_sweep: PdwSweep,
    /// Upper ends of the sweep / map axes (K, Tesla).
    pub t_max: f64,
    pub b_max: f64,
    /// Human-readable system label for the readouts.
    pub system: String,
}

impl PdwRuntime {
    /// |B · u| for the applied field, and whether the field lies along the
    /// sweep axis — only then does the current state sit on the field sweep
    /// and regime map.
    pub fn field_along_axis(&self) -> (f64, bool) {
        let field = self.config.field;
        let along: f64 = field.iter().zip(&self.direction).map(|(b, u)| b * u).sum();
        let norm = (field[0] * field[0] + field[1] * field[1] + field[2] * field[2]).sqrt();
        (along.abs(), (norm - along.abs()).abs() < 1e-9)
    }
}

/// Which plot or field the Spin Fluctuations tab displays. Only the glue map
/// is a moire-grid texture; the transport views live on their own axes (like
/// the PDW sweeps), so they render as plots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SfView {
    /// Δρ/ρ(0) against μ₀H for the 20–50 K family and the current T.
    #[default]
    Magnetoresistance,
    /// dρ/dμ₀H against μ₀H/T, with the hole-doped quadrature form.
    FieldDerivative,
    /// Zero-field ρ(T) and its 15–60 K fit.
    Resistivity,
    /// Jones–Zener R_H(T) with and without vertex corrections.
    Hall,
    /// ρ(0, 40 K) and MR(20 T) across the 19 dopings.
    Doping,
    /// Γ(φ) on the Fermi surface (a side graph on the web page).
    FermiSurface,
    /// SPECULATIVE coupling against Tc with both pairing maps (a side graph
    /// on the web page).
    Coupling,
    /// SPECULATIVE glue imprint against the operating Tc*.
    Imprint,
    /// SPECULATIVE glue-modulated gap Δ(r) on the moire grid.
    GlueGap,
}

impl SfView {
    pub const ALL: [SfView; 9] = [
        SfView::Magnetoresistance,
        SfView::FieldDerivative,
        SfView::Resistivity,
        SfView::Hall,
        SfView::Doping,
        SfView::FermiSurface,
        SfView::Coupling,
        SfView::Imprint,
        SfView::GlueGap,
    ];

    /// The web page's view labels, plus the two side graphs.
    pub fn label(&self) -> &'static str {
        match self {
            SfView::Magnetoresistance => "Magnetoresistance Δρ/ρ(0) vs μ₀H",
            SfView::FieldDerivative => "dρ/dμ₀H vs μ₀H/T (vs hole-doped quadrature)",
            SfView::Resistivity => "Zero-field ρ(T) and its ρ₀ + A₁T + A₂T² fit",
            SfView::Hall => "Hall coefficient R_H(T), with and without vertex corrections",
            SfView::Doping => "Across the dome: ρ(0, 40 K) and MR(20 T) vs x",
            SfView::FermiSurface => "Γ(φ) on the Fermi surface",
            SfView::Coupling => "Coupling λ vs Tc (speculative curves)",
            SfView::Imprint => "Glue route vs amplitude route (speculative)",
            SfView::GlueGap => "Glue-modulated gap Δ(r) (speculative)",
        }
    }

    /// Views drawn on their own axes rather than as moire-grid textures.
    pub fn is_plot(&self) -> bool {
        !matches!(self, SfView::GlueGap)
    }
}

/// User-tunable Spin Fluctuations settings: web bindings 1–16 plus the
/// coherence model (binding 23). The layers, twist and grid come from the
/// main sidebar. The defaults are the web `_BASE_PRESET`.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SfSettings {
    /// Ce doping x; `sf_config` snaps it to the 19 tabulated values.
    pub doping: f64,
    /// Temperature in K.
    pub temperature: f64,
    /// Top of the MR field grid in T.
    pub field_max: f64,
    pub inelastic: SfInelasticForm,
    pub hotspots: bool,
    /// Impurity rate Γ_imp in ps⁻¹.
    pub gamma_imp: f64,
    /// T-linear rate α₁ in ps⁻¹ K⁻¹ (times g(x)).
    pub alpha1: f64,
    /// T² rate α₂ in ps⁻¹ K⁻².
    pub alpha2: f64,
    /// Hot-spot height in ps⁻¹ (the plotted 31.4; Table 2 prints 50).
    pub hotspot_peak: f64,
    /// Hot-spot 1/e half-width in rad.
    pub hotspot_width: f64,
    /// sin^ν exponent of the paper's form (even).
    pub nu: u32,
    /// Hot-spot angle φ_HS in degrees.
    pub phi_hs_deg: f64,
    /// CVC peak ε₀.
    pub cvc_epsilon: f64,
    pub pairing_form: SfPairingForm,
    /// SPECULATIVE glue modulation depth η.
    pub eta: f64,
    /// SPECULATIVE operating Tc* of the glue what-if in K.
    pub tc_target: f64,
    /// SPECULATIVE coherence model of the glue dressing.
    pub coherence: SfCoherence,
}

impl Default for SfSettings {
    fn default() -> Self {
        let config = SfConfig::default();
        let glue = SfGlueConfig::default();
        Self {
            doping: config.doping,
            temperature: 40.0,
            field_max: sf::SF_FIELD_MAX,
            inelastic: config.inelastic,
            hotspots: config.hotspots,
            gamma_imp: config.gamma_imp,
            alpha1: config.alpha1,
            alpha2: config.alpha2,
            hotspot_peak: config.hotspot_peak,
            hotspot_width: config.hotspot_width,
            nu: config.nu,
            phi_hs_deg: config.phi_hs_deg,
            cvc_epsilon: config.cvc_epsilon,
            pairing_form: glue.pairing.form,
            eta: glue.eta,
            tc_target: glue.tc_target,
            coherence: glue.coherence,
        }
    }
}

impl SfSettings {
    // The web `_update_sf` clamps (SF-spec §8.1), shared by `MoireApp::sf_config`
    // and the panel sliders.
    pub const TEMPERATURE: RangeInclusive<f64> = 2.0..=100.0;
    pub const FIELD_MAX: RangeInclusive<f64> = 5.0..=65.0;
    pub const GAMMA_IMP: RangeInclusive<f64> = 0.5..=100.0;
    pub const ALPHA1: RangeInclusive<f64> = 0.0..=3.0;
    pub const ALPHA2: RangeInclusive<f64> = 0.0..=0.05;
    pub const HOTSPOT_PEAK: RangeInclusive<f64> = 0.0..=200.0;
    pub const HOTSPOT_WIDTH: RangeInclusive<f64> = 0.01..=0.3;
    pub const NU: RangeInclusive<u32> = 2..=24;
    pub const PHI_HS: RangeInclusive<f64> = 15.0..=40.0;
    pub const CVC_EPSILON: RangeInclusive<f64> = 0.0..=1.0;
    pub const ETA: RangeInclusive<f64> = 0.0..=0.5;
    /// Also the imprint curve's Tc* axis; below the McMillan ceiling (43.4 K).
    pub const TC_TARGET: RangeInclusive<f64> = 0.5..=25.0;

    // The float sliders' steps (the panel). egui re-snaps a slider's value to
    // its step on every draw, with binary noise (31.4 becomes
    // 31.400000000000002 just by being shown), so `sf_config` rounds each
    // setting to its step's decimals, which keeps the paper's defaults equal
    // to `SfConfig::default()` and the transport cache warm.
    pub const TEMPERATURE_STEP: f64 = 0.5;
    pub const FIELD_MAX_STEP: f64 = 1.0;
    pub const GAMMA_IMP_STEP: f64 = 0.5;
    pub const ALPHA1_STEP: f64 = 0.01;
    pub const ALPHA2_STEP: f64 = 0.0002;
    pub const HOTSPOT_PEAK_STEP: f64 = 0.1;
    pub const HOTSPOT_WIDTH_STEP: f64 = 0.005;
    pub const PHI_HS_STEP: f64 = 0.1;
    pub const CVC_EPSILON_STEP: f64 = 0.01;
    pub const ETA_STEP: f64 = 0.01;
    pub const TC_TARGET_STEP: f64 = 0.1;
}

/// Round `value` to the decimal places of `step` (0.0002 has four), decimal
/// safely: `(x · 10^d).round() / 10^d` returns the literal 31.4 for egui's
/// 31.400000000000002, whereas re-snapping to the step would keep the noise.
/// Values already on the grid are unchanged.
fn snap_to_step(value: f64, step: f64) -> f64 {
    let decimals = (-step.log10()).ceil().max(0.0) as i32;
    let scale = 10f64.powi(decimals);
    (value * scale).round() / scale
}

/// The inputs of one Spin Fluctuations recompute: the settings after the web
/// clamps, plus the glue configs for the selected layers (`MoireApp::sf_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct SfInputs {
    pub config: SfConfig,
    /// Current temperature in K.
    pub temperature: f64,
    /// Top of the MR field grid in T, snapped down to the 0.5 T step.
    pub field_max: f64,
    pub pairing: SfPairingMap,
    /// The glue settings at a placeholder L_m, for the L_m-free readouts
    /// (ξ and Δ̄ do not depend on L_m).
    pub glue_free: SfGlueConfig,
    /// The glue settings at the shared two-lattice period; `None` when the
    /// selected layers have no moire at this twist.
    pub glue: Option<SfGlueConfig>,
    /// Why `glue` is `None` (the PDW wording).
    pub glue_message: Option<String>,
    /// "Sb₂Te₃/FeTe, θ = 0.0°".
    pub system: String,
}

/// Field-max-dependent transport behind the Spin Fluctuations plots, cached on
/// (config, field_max). A rebuild costs about 32 ms single-threaded, or 6 ms
/// on 11 cores (release).
#[derive(Debug, Clone, PartialEq)]
pub struct SfTransport {
    pub config: SfConfig,
    pub field_max: f64,
    /// The MR family at `SF_MR_TEMPERATURES` on 0..=field_max.
    pub mr: SfMagnetoresistance,
    /// Hole-doped quadrature dρ/dμ₀H (Ayres 2021, illustrative Tl2201
    /// parameters) on `mr.fields`, one row per family temperature.
    pub quadrature: Vec<Vec<f64>>,
    /// 2..=100 K at 1 K, and the closed-form ρ(0, T) in μΩ cm.
    pub rho_t: Vec<f64>,
    pub rho: Vec<f64>,
    /// ρ₀ + A₁T + A₂T² over `SF_RHO_FIT_RANGE`.
    pub rho_fit: SfResistivityFit,
    /// 2..=100 K at 2 K, and Jones–Zener R_H(T) in m³/C without and with CVCs.
    pub hall_t: Vec<f64>,
    pub hall_plain: Vec<f64>,
    pub hall_cvc: Vec<f64>,
    /// The 19-doping sweep at 40 K and 20 T.
    pub doping: Vec<SfDopingPoint>,
}

/// SPECULATIVE glue imprint against the operating Tc* (`imprint_curve`).
#[derive(Debug, Clone, PartialEq)]
pub struct SfImprint {
    pub tc_targets: Vec<f64>,
    pub dressed: Vec<f64>,
    pub local: Vec<f64>,
    pub linear: Vec<f64>,
    /// Upper end of the "Tc(r) = 0 in places" shading: set only in the local
    /// limit, and only when the map fragments above the slider minimum.
    pub fragment_below: Option<f64>,
}

/// SPECULATIVE Tc(λ) of both pairing maps on one λ grid (the coupling plot).
#[derive(Debug, Clone, PartialEq)]
pub struct SfPairingCurves {
    pub lambda: Vec<f64>,
    pub power_law: Vec<f64>,
    pub mcmillan: Vec<f64>,
}

/// Scalars for the info panel and the sidebar readout, at the current T
/// unless named. All are direct evaluations, independent of field_max. The six
/// L_m-dependent glue readouts are `None` when the layers have no moire.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfReadouts {
    pub g: f64,
    /// Table Tc(x) in K.
    pub tc: f64,
    pub lambda_tr: f64,
    pub lambda_rho: f64,
    /// ρ(0, T) in μΩ cm.
    pub rho0: f64,
    /// MR(20 T) at T.
    pub mr_20t: f64,
    /// γ₁ = dρ/dμ₀H at 33 T in μΩ cm/T.
    pub slope_33t: f64,
    /// R_H in m³/C: without and with CVCs, Drude on the fixed x = 0.159 FS,
    /// and Drude for n = 1 − x.
    pub rh_plain: f64,
    pub rh_cvc: f64,
    pub rh_drude: f64,
    pub rh_drude_nominal: f64,
    /// R_H / R_H(case a) with CVCs: (d)/(a) for the paper's form.
    pub rh_ratio_to_case_a: f64,
    /// The Kohler pair: MR(20 T) at 20 K and at 50 K.
    pub mr_20k_20t: f64,
    pub mr_50k_20t: f64,
    /// MR(50 K, 20 T) over MR(20 K) at the same B/ρ₀, i.e. at
    /// 20 T · ρ₀(20 K)/ρ₀(50 K): 1 under Kohler's rule (the web
    /// `_kohler_ratio`; 1 by convention when both MRs are 0).
    pub kohler_ratio: f64,
    /// FS × AFM-zone-boundary crossing in degrees.
    pub crossing_deg: f64,
    pub regime: SfDopingRegime,
    /// SPECULATIVE Tc(λ_tr) from the selected pairing map, K.
    pub tc_map_at_x: f64,
    pub lambda_threshold: f64,
    /// The LCCO doping whose λ_tr equals the threshold (None for McMillan).
    pub x_threshold: Option<f64>,
    /// SPECULATIVE glue readouts at the operating Tc*.
    pub lambda_bar: f64,
    pub sensitivity: f64,
    pub x_equiv: Option<f64>,
    /// Δ̄ in meV.
    pub delta_bar: f64,
    /// ξ(Tc*) in Å.
    pub coherence_length: f64,
    /// L_m in Å.
    pub moire_period: Option<f64>,
    pub xi_over_lm: Option<f64>,
    pub coherence_factor: Option<f64>,
    /// δΔ/Δ̄: dressed (F × local exact), local exact and linear η·S.
    pub glue_rel_dressed: Option<f64>,
    pub glue_rel_local: f64,
    pub glue_rel_linear: f64,
    /// Share of the map with Tc(r) = 0 in the local limit, and the normal
    /// share of the selected coherence model's map (0 unless local: F < 1
    /// keeps Δ > 0).
    pub local_normal_fraction: Option<f64>,
    pub normal_fraction: Option<f64>,
}

/// Cached spin-fluctuation computation behind the Spin Fluctuations tab.
pub struct SfRuntime {
    /// Reused while (config, field_max) is unchanged.
    pub transport: Arc<SfTransport>,
    /// Current temperature in K.
    pub temperature: f64,
    pub pairing: SfPairingMap,
    /// The clamped glue settings at a placeholder L_m.
    pub glue_free: SfGlueConfig,
    /// The glue settings at the shared period; `None` with no moire.
    pub glue_config: Option<SfGlueConfig>,
    pub fermi: SfFermiSurface,
    /// Γ(φ) on `fermi.phi` at the current T, in ps⁻¹.
    pub gamma: Vec<f64>,
    /// Γ_HS centres, the CVC centres (the FS × zone-boundary crossings) and
    /// their k + Q partners, in rad.
    pub hotspots: [f64; 8],
    pub cvc_centres: [f64; 8],
    pub cvc_partners: [f64; 8],
    /// The MR curve at the current T.
    pub current: SfMagnetoresistance,
    pub pairing_curves: SfPairingCurves,
    /// `None` with no moire.
    pub imprint: Option<SfImprint>,
    /// The dressed glue map on the moire grid; `None` with no moire.
    pub glue: Option<SfGlueResult>,
    /// Resolution and full extent (Å) of the grid behind `glue`.
    pub glue_resolution: usize,
    pub glue_extent: f64,
    /// Why `glue` and `imprint` are missing.
    pub glue_message: Option<String>,
    /// "Sb₂Te₃/FeTe, θ = 0.0°".
    pub system: String,
    pub readouts: SfReadouts,
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

    // --- Spin fluctuations (speculative) ---
    /// User-tunable spin-fluctuation settings.
    #[serde(default)]
    pub sf_settings: SfSettings,
    /// Which plot or field the Spin Fluctuations tab displays.
    #[serde(default)]
    pub sf_view: SfView,

    // --- Pair density wave (speculative) ---
    /// User-tunable PDW settings.
    #[serde(default)]
    pub pdw_settings: PdwSettings,
    /// Which field or plot the Pair Density Wave tab displays.
    #[serde(default)]
    pub pdw_view: PdwView,

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

    // --- Pair density wave runtime state (not serialized) ---
    /// Cached PDW computation.
    #[serde(skip)]
    pub pdw_result: Option<PdwRuntime>,
    /// Texture for the active PDW view (field map, FFT or regime map).
    #[serde(skip)]
    pub pdw_texture: Option<egui::TextureHandle>,
    /// Theme the regime-map texture was painted for (its colours are themed).
    #[serde(skip)]
    pub pdw_texture_dark: Option<bool>,
    /// Why the last PDW computation failed (shown in place of its views).
    #[serde(skip)]
    pub pdw_error: Option<String>,
    /// Whether the PDW stage needs recompute.
    #[serde(skip)]
    pub needs_pdw_recompute: bool,

    // --- Spin fluctuations runtime state (not serialized) ---
    /// Cached spin-fluctuation computation.
    #[serde(skip)]
    pub sf_result: Option<SfRuntime>,
    /// Texture of the glue map (the only SF texture view).
    #[serde(skip)]
    pub sf_texture: Option<egui::TextureHandle>,
    /// Why the last SF computation failed (shown in place of its views).
    #[serde(skip)]
    pub sf_error: Option<String>,
    /// Whether the SF stage is dirty. It recomputes only while the Spin
    /// Fluctuations tab is open; elsewhere the flag stays set.
    #[serde(skip)]
    pub needs_sf_recompute: bool,
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
            sf_settings: SfSettings::default(),
            sf_view: SfView::default(),
            pdw_settings: PdwSettings::default(),
            pdw_view: PdwView::default(),
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
            pdw_result: None,
            pdw_texture: None,
            pdw_texture_dark: None,
            pdw_error: None,
            needs_pdw_recompute: true,
            sf_result: None,
            sf_texture: None,
            sf_error: None,
            needs_sf_recompute: true,
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
    /// without touching on-screen textures. Public so headless drivers (the
    /// docs capture tool) can grab exactly what the app shows; `None` for plot
    /// views and before anything is computed.
    pub fn capture_current_view(&self) -> Option<egui::ColorImage> {
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
            Tab::SpinFluctuation => {
                let (data, cmap) = self.sf_view_scalar()?;
                let n = self.sf_result.as_ref()?.glue_resolution;
                (data, n, cmap)
            }
            Tab::PairDensityWave => {
                let (data, cmap) = self.pdw_view_scalar()?;
                let n = self.pdw_result.as_ref()?.fields.resolution;
                (data, n, cmap)
            }
        };

        // 2D: colormap the grid directly at the source resolution, with data
        // row 0 (y_min) at the bottom as the viewport's flipped uv shows it.
        // (We keep the raw grid resolution rather than interpolating to 1024².)
        let pixels: Vec<egui::Color32> = data_vec
            .chunks_exact(n)
            .rev()
            .flatten()
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
                    && self.cooper_view == CooperView::DecayProfile)
                || (self.active_tab == Tab::PairDensityWave && self.pdw_view.is_plot())
                || (self.active_tab == Tab::SpinFluctuation && self.sf_view.is_plot());
            let msg = if is_plot_view {
                "Screenshot skipped: band/DOS/decay/PDW-sweep/SF plots are not captured"
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
            Tab::SpinFluctuation => {
                // Only the glue map has a surface; the plot views return None.
                let (data, cmap) = self.sf_view_scalar()?;
                let n = self.sf_result.as_ref()?.glue_resolution;
                Some(SurfaceScalar {
                    heights: data,
                    colors: None,
                    n,
                    colormap: cmap,
                })
            }
            Tab::PairDensityWave => {
                let (data, cmap) = self.pdw_view_scalar()?;
                let n = self.pdw_result.as_ref()?.fields.resolution;
                Some(SurfaceScalar {
                    heights: data,
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

    /// Assemble the PDW model for the current settings, plus a system label.
    /// The moire-locked geometry uses the shared two-lattice period and
    /// two-gap amplitude (`moire_locked_period`, `moire_locked_pdw_ratio`) so
    /// it matches the web page at any twist, rather than the moire stage's
    /// own period estimate.
    fn pdw_config(&self) -> Result<(PdwConfig, String), String> {
        let s = &self.pdw_settings;
        let (base, system) = match s.geometry {
            PdwGeometry::Ute2Surface => (PdwConfig::default(), "UTe2 (01-1) surface".to_string()),
            PdwGeometry::MoireLocked => {
                let (substrate, overlayer) = (self.substrate_material(), self.overlayer_material());
                let period = moire_core::pdw::moire_locked_period(
                    substrate.a,
                    overlayer.a,
                    self.twist_angle,
                )?;
                if !period.is_finite() {
                    return Err(format!(
                        "{}/{} has no moire at this twist; pick different layers or a nonzero twist",
                        overlayer.formula, substrate.formula
                    ));
                }
                let (delta_1, delta_2) = (self.density_config.delta_1, self.density_config.delta_2);
                let ratio = moire_core::pdw::moire_locked_pdw_ratio(
                    period,
                    DEFAULT_COHERENCE_LENGTH,
                    delta_1,
                    delta_2,
                )?;
                let delta_avg = (delta_1 + delta_2) / 2.0;
                let cfg = PdwConfig::moire_locked(period, self.twist_angle, delta_avg, ratio)?;
                let label = format!(
                    "{}/{}, L_m = {:.1} Å",
                    overlayer.formula, substrate.formula, period
                );
                (cfg, label)
            }
        };
        let config = PdwConfig {
            temperature: s.temperature,
            t_pdw: s.tpdw_ratio * base.tc,
            field: s.field,
            kappa_p: s.kappa_p,
            kappa_q: s.kappa_q,
            kappa_h: s.kappa_h,
            kappa_harm: s.kappa_harm,
            floor: 10f64.powf(s.floor_log10),
            ..base
        };
        Ok((config, system))
    }

    /// Run the PDW model (state, real-space fields, both sweeps) and rebuild
    /// the active-view texture. Sweeps are a few hundred closed-form states,
    /// so they are always refreshed; the FFT only while its view is active.
    fn recompute_pdw(&mut self, ctx: &egui::Context) {
        let n = self.resolution;
        let extent = self.physical_extent;
        let want_fft = self.pdw_view == PdwView::Fourier;
        let built = self.pdw_config().and_then(|(config, system)| {
            use moire_core::pdw;
            let state = pdw::compute_pdw_state(&config)?;
            let fields = pdw::pdw_real_space(&config, &state, n, extent)?;
            let fft = if want_fft {
                Some(moire_core::fft::compute_fft_2d(
                    &hann_taper(&fields.charge_density, n),
                    n,
                )?)
            } else {
                None
            };
            let direction = self.pdw_settings.sweep_axis.direction(config.field);
            let along = config
                .field
                .iter()
                .zip(&direction)
                .map(|(b, u)| b * u)
                .sum::<f64>();
            let t_max = PDW_T_RANGE_FACTOR * config.tc.max(config.t_pdw);
            let b_max = PDW_B_RANGE_FACTOR
                * pdw::critical_field_along(config.tc, &config.hc2, 0.0, &direction)?
                    .max(pdw::critical_field_along(
                        config.t_pdw,
                        &config.h_pdw,
                        0.0,
                        &direction,
                    )?)
                    .max(along.abs());
            let t_sweep = pdw::temperature_sweep(&config, &linspace(0.0, t_max, PDW_SWEEP_POINTS))?;
            let b_sweep =
                pdw::field_sweep(&config, &linspace(0.0, b_max, PDW_SWEEP_POINTS), &direction)?;
            Ok(PdwRuntime {
                peaks: pdw::pdw_peaks(&config, &state),
                config,
                state,
                fields,
                fft,
                direction,
                t_sweep,
                b_sweep,
                t_max,
                b_max,
                system,
            })
        });
        // Record the theme either way, so the regime-map re-theme check in
        // `update` cannot re-trigger a failing computation every frame.
        self.pdw_texture_dark = Some(self.dark_mode);
        match built {
            Ok(rt) => {
                self.pdw_result = Some(rt);
                self.pdw_error = None;
            }
            Err(e) => {
                eprintln!("PDW computation error: {e}");
                self.pdw_result = None;
                self.pdw_texture = None;
                self.pdw_error = Some(e);
                return;
            }
        }

        self.pdw_texture = if self.pdw_view == PdwView::RegimeMap {
            self.pdw_regime_image()
                .map(|img| ctx.load_texture("pdw_regimes", img, egui::TextureOptions::NEAREST))
        } else {
            self.pdw_view_scalar().map(|(data, cmap)| {
                render::pattern::create_texture(ctx, "pdw_view", &data, n, cmap)
            })
        };
        self.needs_surface_rerender = true;
    }

    /// Scalar field for the active PDW view, normalized to [0, 1], plus its
    /// colormap. Signed densities map symmetrically about 0.5; the plot views
    /// return `None`.
    #[allow(clippy::type_complexity)]
    fn pdw_view_scalar(&self) -> Option<(Vec<f64>, fn(f64) -> [u8; 4])> {
        let rt = self.pdw_result.as_ref()?;
        let coolwarm = self.cmap(moire_core::colormap::coolwarm);
        match self.pdw_view {
            PdwView::PairAmplitude => {
                let pair = &rt.fields.pair_amplitude;
                let norm = normalize_to_unit_range(pair).unwrap_or_else(|| vec![0.5; pair.len()]);
                Some((norm, coolwarm))
            }
            PdwView::ChargeDensity => {
                Some((symmetric_unit_range(&rt.fields.charge_density), coolwarm))
            }
            PdwView::SpinDensity => Some((symmetric_unit_range(&rt.fields.spin_density), coolwarm)),
            // Already log-scaled to [0, 1]; same colormap as the Fourier tab.
            PdwView::Fourier => {
                let fft = rt.fft.as_ref()?;
                Some((fft.clone(), self.cmap(moire_core::colormap::inferno)))
            }
            PdwView::TemperatureSweep | PdwView::FieldSweep | PdwView::RegimeMap => None,
        }
    }

    /// Regime map over (|B| along the sweep axis, T) as a themed image with T
    /// rising upward, from `moire_core::pdw::regime_map`.
    pub fn pdw_regime_image(&self) -> Option<egui::ColorImage> {
        let rt = self.pdw_result.as_ref()?;
        let n = PDW_REGIME_N;
        let temps = linspace(0.0, rt.t_max, n);
        let mags = linspace(0.0, rt.b_max, n);
        let map = moire_core::pdw::regime_map(&rt.config, &temps, &mags, &rt.direction).ok()?;
        let mut pixels = Vec::with_capacity(n * n);
        for row in 0..n {
            let t_idx = n - 1 - row;
            for b_idx in 0..n {
                let regime: PdwRegime = map[t_idx * n + b_idx];
                pixels.push(ui::pdw_panel::regime_color(regime, self.dark_mode));
            }
        }
        Some(egui::ColorImage {
            size: [n, n],
            pixels,
        })
    }

    /// Assemble the spin-fluctuation inputs from the settings with the web
    /// `_update_sf` clamps (snapped doping, even ν, field max on the 0.5 T
    /// step). Unlike the web `_number` fallback it rejects non-finite settings
    /// (`f64::clamp` passes NaN through), which makes it the tab's error path.
    /// L_m is the shared two-lattice period of the selected layers, as for the
    /// moire-locked PDW; without a moire the glue config is `None` and the
    /// reason travels with the inputs instead of failing the stage.
    fn sf_config(&self) -> Result<SfInputs, String> {
        let s = &self.sf_settings;
        for (name, value) in [
            ("doping", s.doping),
            ("temperature", s.temperature),
            ("field_max", s.field_max),
            ("gamma_imp", s.gamma_imp),
            ("alpha1", s.alpha1),
            ("alpha2", s.alpha2),
            ("hotspot_peak", s.hotspot_peak),
            ("hotspot_width", s.hotspot_width),
            ("phi_hs_deg", s.phi_hs_deg),
            ("cvc_epsilon", s.cvc_epsilon),
            ("eta", s.eta),
            ("tc_target", s.tc_target),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite, got {value}"));
            }
        }
        // Clamp to the web range, then drop the slider's binary noise (the
        // range ends are exact at the step's decimals, so the value stays
        // inside the range).
        let clamp = |value: f64, range: RangeInclusive<f64>, step: f64| {
            snap_to_step(value.clamp(*range.start(), *range.end()), step)
        };
        // Nearest even integer with ties down (13 -> 12): the web 2·ceil(ν/2 − ½).
        let (nu_min, nu_max) = (*SfSettings::NU.start(), *SfSettings::NU.end());
        let nu = (2.0 * (f64::from(s.nu) / 2.0 - 0.5).ceil())
            .clamp(f64::from(nu_min), f64::from(nu_max)) as u32;
        // field_max is not rounded to its 1 T slider step: the web floors it
        // to the 0.5 T grid (35.74 -> 35.5), and integers carry no noise.
        let field_max = (s
            .field_max
            .clamp(*SfSettings::FIELD_MAX.start(), *SfSettings::FIELD_MAX.end())
            / sf::SF_FIELD_STEP)
            .floor()
            * sf::SF_FIELD_STEP;
        let config = SfConfig {
            doping: sf::snap_doping(s.doping),
            inelastic: s.inelastic,
            hotspots: s.hotspots,
            gamma_imp: clamp(
                s.gamma_imp,
                SfSettings::GAMMA_IMP,
                SfSettings::GAMMA_IMP_STEP,
            ),
            alpha1: clamp(s.alpha1, SfSettings::ALPHA1, SfSettings::ALPHA1_STEP),
            alpha2: clamp(s.alpha2, SfSettings::ALPHA2, SfSettings::ALPHA2_STEP),
            hotspot_peak: clamp(
                s.hotspot_peak,
                SfSettings::HOTSPOT_PEAK,
                SfSettings::HOTSPOT_PEAK_STEP,
            ),
            hotspot_width: clamp(
                s.hotspot_width,
                SfSettings::HOTSPOT_WIDTH,
                SfSettings::HOTSPOT_WIDTH_STEP,
            ),
            nu,
            phi_hs_deg: clamp(s.phi_hs_deg, SfSettings::PHI_HS, SfSettings::PHI_HS_STEP),
            cvc_epsilon: clamp(
                s.cvc_epsilon,
                SfSettings::CVC_EPSILON,
                SfSettings::CVC_EPSILON_STEP,
            ),
            ..SfConfig::default()
        };
        let pairing = SfPairingMap {
            form: s.pairing_form,
            ..SfPairingMap::default()
        };
        // The amplitude-route constants come from the core (I7): Δ̄ scales the
        // core DELTA_AVG, never the sidebar Δ₁ / Δ₂.
        let glue_free = SfGlueConfig {
            eta: clamp(s.eta, SfSettings::ETA, SfSettings::ETA_STEP),
            tc_target: clamp(
                s.tc_target,
                SfSettings::TC_TARGET,
                SfSettings::TC_TARGET_STEP,
            ),
            coherence: s.coherence,
            pairing,
            ..SfGlueConfig::default()
        };
        let (substrate, overlayer) = (self.substrate_material(), self.overlayer_material());
        let period =
            moire_core::pdw::moire_locked_period(substrate.a, overlayer.a, self.twist_angle)?;
        let (glue, glue_message) = if period.is_finite() {
            let glue = SfGlueConfig {
                moire_period: period,
                ..glue_free
            };
            (Some(glue), None)
        } else {
            let message = format!(
                "{}/{} has no moire at this twist; pick different layers or a nonzero twist",
                overlayer.formula, substrate.formula
            );
            (None, Some(message))
        };
        Ok(SfInputs {
            config,
            temperature: clamp(
                s.temperature,
                SfSettings::TEMPERATURE,
                SfSettings::TEMPERATURE_STEP,
            ),
            field_max,
            pairing,
            glue_free,
            glue,
            glue_message,
            system: format!(
                "{}/{}, θ = {:.1}°",
                overlayer.formula, substrate.formula, self.twist_angle
            ),
        })
    }

    /// Run the spin-fluctuation stage. The transport bundle is reused while
    /// (config, field_max) is unchanged; Γ(φ), the current-T MR curve, the
    /// readouts, the pairing curves, the imprint curve and the glue map are
    /// rebuilt on every call. A failure clears the stage and keeps the message
    /// for the viewport, as `recompute_pdw` does.
    fn recompute_sf(&mut self, ctx: &egui::Context) {
        let built = self
            .sf_config()
            .and_then(|inputs| self.build_sf_runtime(inputs));
        match built {
            Ok(rt) => {
                self.sf_result = Some(rt);
                self.sf_error = None;
            }
            Err(e) => {
                eprintln!("Spin-fluctuation computation error: {e}");
                self.sf_result = None;
                self.sf_texture = None;
                self.sf_error = Some(e);
                return;
            }
        }

        // Only the glue map is a texture; the other views are plots.
        self.sf_texture = match (self.sf_view_scalar(), self.sf_result.as_ref()) {
            (Some((data, cmap)), Some(rt)) => Some(render::pattern::create_texture(
                ctx,
                "sf_view",
                &data,
                rt.glue_resolution,
                cmap,
            )),
            _ => None,
        };
        self.needs_surface_rerender = true;
    }

    /// Build the SF runtime for clamped inputs, reusing the cached transport
    /// bundle when its (config, field_max) key matches.
    fn build_sf_runtime(&self, inputs: SfInputs) -> Result<SfRuntime, String> {
        let SfInputs {
            config,
            temperature,
            field_max,
            pairing,
            glue_free,
            glue,
            glue_message,
            system,
        } = inputs;
        let transport = match self.sf_result.as_ref() {
            Some(rt) if rt.transport.config == config && rt.transport.field_max == field_max => {
                Arc::clone(&rt.transport)
            }
            _ => Arc::new(sf_transport(&config, field_max)?),
        };
        let current = sf::magnetoresistance(&config, &[temperature], field_max, sf::SF_FIELD_STEP)?;
        let fermi = sf::fermi_surface(&config)?;
        let gamma = fermi
            .phi
            .iter()
            .map(|&phi| sf::scattering_rate(phi, temperature, &config))
            .collect::<Result<Vec<_>, _>>()?;
        let cvc_centres = sf::cvc_centres(&config)?;
        let cvc_partners = cvc_centres.map(|c| sf::cvc_partner_angle(c, &config));

        // The glue map reads the moire pattern; the imprint is pattern-free.
        let (glue_map, imprint) = match glue.as_ref() {
            Some(g) => {
                let (tc_min, tc_max) =
                    (*SfSettings::TC_TARGET.start(), *SfSettings::TC_TARGET.end());
                let tc_targets = linspace(tc_min, tc_max, SF_IMPRINT_POINTS);
                let (dressed, local, linear) = sf::imprint_curve(g, &tc_targets)?;
                // Shade only in the local limit, and only when the map fragments
                // above the slider minimum (McMillan never fragments).
                let fragment_below = if g.coherence == SfCoherence::Local {
                    let tc_frag = sf::fragmentation_tc(g)?;
                    (tc_frag > tc_min).then_some(tc_frag)
                } else {
                    None
                };
                let map = self
                    .moire_result
                    .as_ref()
                    .map(|m| {
                        sf::glue_modulated_gap(&m.pattern, g)
                            .map(|r| (r, m.resolution, m.physical_extent))
                    })
                    .transpose()?;
                let imprint = SfImprint {
                    tc_targets,
                    dressed,
                    local,
                    linear,
                    fragment_below,
                };
                (map, Some(imprint))
            }
            None => (None, None),
        };
        let readouts = sf_readouts(
            &config,
            temperature,
            &pairing,
            &glue_free,
            glue.as_ref(),
            glue_map.as_ref().map(|(map, _, _)| map),
        )?;
        let (glue_map, glue_resolution, glue_extent) = match glue_map {
            Some((map, n, extent)) => (Some(map), n, extent),
            None => (None, 0, 0.0),
        };
        Ok(SfRuntime {
            transport,
            temperature,
            pairing,
            glue_free,
            glue_config: glue,
            hotspots: sf::hotspot_angles(&config),
            cvc_centres,
            cvc_partners,
            fermi,
            gamma,
            current,
            pairing_curves: sf_pairing_curves(),
            imprint,
            glue: glue_map,
            glue_resolution,
            glue_extent,
            glue_message,
            system,
            readouts,
        })
    }

    /// Scalar field for the active SF view: the dressed glue map normalized to
    /// [0, 1] with coolwarm ("red = larger Δ"), exactly as `pdw_view_scalar`
    /// treats |d(r)|. A uniform map (η = 0) sits at 0.5 rather than going
    /// blank; the plot views return `None`.
    #[allow(clippy::type_complexity)]
    fn sf_view_scalar(&self) -> Option<(Vec<f64>, fn(f64) -> [u8; 4])> {
        if self.sf_view != SfView::GlueGap {
            return None;
        }
        let gap = &self.sf_result.as_ref()?.glue.as_ref()?.gap;
        let norm = normalize_to_unit_range(gap).unwrap_or_else(|| vec![0.5; gap.len()]);
        Some((norm, self.cmap(moire_core::colormap::coolwarm)))
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

    /// The compute pass of `update()`: the main pipeline, every stage that
    /// depends on it, then the 3D re-render. Public so headless drivers run the
    /// same pass as the window.
    pub fn run_compute_stages(&mut self, ctx: &egui::Context) {
        if self.needs_recompute {
            self.recompute(ctx);
            self.needs_magnetic_recompute = true;
            self.needs_graphene_recompute = true;
            // The moire-locked PDW reads the moire period; the field maps share
            // the resolution, extent and colormap override.
            self.needs_pdw_recompute = true;
            // The glue map reads the moire pattern and the colormap override.
            self.needs_sf_recompute = true;
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

        // The regime-map texture is painted in themed colours.
        if self.pdw_view == PdwView::RegimeMap && self.pdw_texture_dark != Some(self.dark_mode) {
            self.needs_pdw_recompute = true;
        }
        if self.needs_pdw_recompute {
            self.recompute_pdw(ctx);
            self.needs_pdw_recompute = false;
        }

        // The SF transport is the costliest stage, so it runs only while its
        // tab is open; elsewhere the flag stays set until the tab opens.
        if self.needs_sf_recompute && self.active_tab == Tab::SpinFluctuation {
            self.recompute_sf(ctx);
            self.needs_sf_recompute = false;
        }

        if self.needs_surface_rerender && self.view_mode == ViewMode::Surface3D {
            self.rerender_surface(ctx);
            self.needs_surface_rerender = false;
        }
    }

    /// The window's two panels: the controls and results sidebar, and the
    /// viewport with its tab bar.
    pub fn show_panels(&mut self, ctx: &egui::Context) {
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
    }
}

/// The transport bundle of the Spin Fluctuations tab for one (config,
/// field_max) key: the MR family with the quadrature contrast, ρ(0, T) and its
/// fit, R_H(T) without and with CVCs, and the doping sweep.
fn sf_transport(config: &SfConfig, field_max: f64) -> Result<SfTransport, String> {
    let mr = sf::magnetoresistance(
        config,
        &sf::SF_MR_TEMPERATURES,
        field_max,
        sf::SF_FIELD_STEP,
    )?;
    let quad = sf::SfQuadratureConfig::default();
    let quadrature = mr
        .temperatures
        .iter()
        .map(|&t| {
            mr.fields
                .iter()
                .map(|&b| sf::quadrature_field_derivative(t, b, &quad))
                .collect()
        })
        .collect();
    let rho_t = linspace(2.0, 100.0, SF_RHO_T_POINTS);
    let rho = sf::zero_field_resistivity(config, &rho_t)?;
    let (fit_min, fit_max) = sf::SF_RHO_FIT_RANGE;
    let rho_fit = sf::resistivity_fit(config, fit_min, fit_max, 1.0)?;
    let hall_t = linspace(2.0, 100.0, SF_HALL_T_POINTS);
    Ok(SfTransport {
        hall_plain: sf::hall_coefficient_curve(config, &hall_t, false)?,
        hall_cvc: sf::hall_coefficient_curve(config, &hall_t, true)?,
        doping: sf::doping_sweep(config, sf::SF_FIT_TEMPERATURE, sf::SF_FIT_FIELD)?,
        config: *config,
        field_max,
        mr,
        quadrature,
        rho_t,
        rho,
        rho_fit,
        hall_t,
    })
}

/// The SF readouts at temperature `t`: direct evaluations, independent of the
/// MR field grid (I4). The L_m-dependent glue values need `glue`, and the
/// normal fractions need the glue map.
fn sf_readouts(
    config: &SfConfig,
    t: f64,
    pairing: &SfPairingMap,
    glue_free: &SfGlueConfig,
    glue: Option<&SfGlueConfig>,
    glue_map: Option<&SfGlueResult>,
) -> Result<SfReadouts, String> {
    /// MR(20 T) of the readouts and the Kohler pair.
    const MR_FIELD: f64 = 20.0;
    let mr_at = |temperature: f64, field: f64| -> Result<f64, String> {
        let r = sf::resistivity_tensor(config, temperature, &[0.0, field])?;
        Ok(r[1][0] / r[0][0] - 1.0)
    };
    // Kohler's rule makes MR a function of B/ρ₀ alone, so the 50 K value is
    // compared with MR(20 K) at the field 20 T · ρ₀(20 K)/ρ₀(50 K), as the web
    // `_kohler_ratio` does. A fall of MR(20 T) from 20 K to 50 K alone is not
    // Kohler behaviour: ρ₀ grows with T, so the rule predicts the fall and its
    // size (a uniform Γ collapses exactly).
    let (mr_20k_20t, mr_50k_20t) = (mr_at(20.0, MR_FIELD)?, mr_at(50.0, MR_FIELD)?);
    let rho = sf::zero_field_resistivity(config, &[20.0, 50.0])?;
    let reference = mr_at(20.0, MR_FIELD * rho[0] / rho[1])?;
    let kohler_ratio = if reference == 0.0 {
        // No MR at all at 20 K (not reached on the sliders' ranges).
        if mr_50k_20t == 0.0 {
            1.0
        } else {
            f64::INFINITY
        }
    } else {
        mr_50k_20t / reference
    };
    let lambda_tr = sf::lambda_tr(config)?;
    let lambda_threshold = sf::lambda_threshold(pairing);
    let lambda_bar = sf::lambda_from_tc(glue_free.tc_target, pairing)?;
    let (glue_rel_local, glue_rel_linear) =
        sf::local_imprint(glue_free.eta, glue_free.tc_target, pairing)?;
    // ξ and Δ̄ do not depend on L_m, so the placeholder period serves.
    let coherence_length = sf::coherence_length(glue_free)?;
    let moire_period = glue.map(|g| g.moire_period);
    let coherence_factor = glue.map(sf::coherence_factor).transpose()?;
    Ok(SfReadouts {
        g: sf::g_of_x(config.doping)?,
        tc: sf::tc_of_x(config.doping),
        lambda_tr,
        lambda_rho: sf::lambda_resistive(config)?,
        rho0: sf::zero_field_resistivity(config, &[t])?[0],
        mr_20t: mr_at(t, MR_FIELD)?,
        slope_33t: sf::high_field_slope(config, t, sf::SF_SLOPE_FIELD, sf::SF_FIELD_STEP)?,
        rh_plain: sf::hall_coefficient(config, t, false)?,
        rh_cvc: sf::hall_coefficient(config, t, true)?,
        rh_drude: sf::drude_hall_coefficient(config)?,
        rh_drude_nominal: sf::drude_hall_coefficient_nominal(config)?,
        rh_ratio_to_case_a: sf::hall_ratio_to_case_a(config, t)?,
        mr_20k_20t,
        mr_50k_20t,
        kohler_ratio,
        crossing_deg: sf::zone_boundary_crossing_deg(config)?,
        regime: sf::doping_regime(config.doping),
        tc_map_at_x: sf::tc_from_lambda(lambda_tr, pairing),
        lambda_threshold,
        x_threshold: sf::equivalent_doping(lambda_threshold),
        lambda_bar,
        sensitivity: sf::log_sensitivity(lambda_bar, pairing),
        x_equiv: sf::equivalent_doping(lambda_bar),
        delta_bar: sf::glue_gap_scale(glue_free)?,
        coherence_length,
        moire_period,
        xi_over_lm: moire_period.map(|lm| coherence_length / lm),
        coherence_factor,
        glue_rel_dressed: coherence_factor.map(|f| f * glue_rel_local),
        glue_rel_local,
        glue_rel_linear,
        local_normal_fraction: glue_map.map(|m| m.local_normal_fraction),
        normal_fraction: glue_map.map(|m| m.normal_fraction),
    })
}

/// SPECULATIVE Tc(λ) of both pairing maps on 0..=SF_PAIRING_LAMBDA_MAX, for
/// the coupling plot.
fn sf_pairing_curves() -> SfPairingCurves {
    let lambda = linspace(0.0, SF_PAIRING_LAMBDA_MAX, SF_PAIRING_CURVE_POINTS);
    let curve = |form: SfPairingForm| -> Vec<f64> {
        let map = SfPairingMap {
            form,
            ..SfPairingMap::default()
        };
        lambda
            .iter()
            .map(|&l| sf::tc_from_lambda(l, &map))
            .collect()
    };
    let power_law = curve(SfPairingForm::PowerLaw);
    let mcmillan = curve(SfPairingForm::McMillan);
    SfPairingCurves {
        lambda,
        power_law,
        mcmillan,
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

        self.run_compute_stages(ctx);
        self.show_panels(ctx);

        // Opening the Spin Fluctuations tab (or editing it) leaves its stage
        // dirty until the next compute pass: ask for that frame now instead of
        // waiting for the next input event.
        if self.active_tab == Tab::SpinFluctuation && self.needs_sf_recompute {
            ctx.request_repaint();
        }

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
                                    // Data row 0 is y_min: v flipped, as in
                                    // the main 2D view.
                                    let uv = egui::Rect::from_min_max(
                                        egui::pos2(0.0, 1.0),
                                        egui::pos2(1.0, 0.0),
                                    );

                                    if moire_idx < textures.len() {
                                        ui.label("Moire Pattern:");
                                        ui.add(
                                            egui::Image::new(egui::load::SizedTexture::new(
                                                textures[moire_idx].id(),
                                                tex_size,
                                            ))
                                            .uv(uv),
                                        );
                                    }
                                    if density_idx < textures.len() {
                                        ui.label("Density:");
                                        ui.add(
                                            egui::Image::new(egui::load::SizedTexture::new(
                                                textures[density_idx].id(),
                                                tex_size,
                                            ))
                                            .uv(uv),
                                        );
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
        app.recompute_pdw(&ctx);
        (app, ctx)
    }

    /// `pipeline_app` plus the SF stage, which `pipeline_app` leaves dirty:
    /// it only runs while its tab is open (I2), so open the tab first.
    fn sf_app(resolution: usize) -> (MoireApp, egui::Context) {
        let (mut app, ctx) = pipeline_app(resolution);
        app.active_tab = Tab::SpinFluctuation;
        app.recompute_sf(&ctx);
        (app, ctx)
    }

    /// Index of `formula` in a material list.
    fn material_index(list: &[&materials::Material], formula: &str) -> usize {
        list.iter()
            .position(|m| m.formula == formula)
            .expect("material in the registry")
    }

    /// Run one headless frame of `size` points and return every piece of
    /// painted text with its screen rect. Off-screen widgets paint nothing.
    fn frame_text(
        ctx: &egui::Context,
        size: egui::Vec2,
        add_contents: impl FnMut(&egui::Context),
    ) -> Vec<(String, egui::Rect)> {
        frame_text_with_events(ctx, size, Vec::new(), add_contents)
    }

    /// Click the first painted text equal to `label`, as a user opening a
    /// collapsing header does: a frame to find it, one to press, one to
    /// release.
    fn click_text(
        ctx: &egui::Context,
        size: egui::Vec2,
        label: &str,
        mut add_contents: impl FnMut(&egui::Context),
    ) {
        let text = frame_text(ctx, size, &mut add_contents);
        let pos = text
            .iter()
            .find(|(t, _)| t == label)
            .map(|(_, rect)| rect.center())
            .unwrap_or_else(|| panic!("{label:?} was not painted"));
        let button = |pressed| egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: egui::Modifiers::NONE,
        };
        frame_text_with_events(
            ctx,
            size,
            vec![egui::Event::PointerMoved(pos), button(true)],
            &mut add_contents,
        );
        frame_text_with_events(ctx, size, vec![button(false)], &mut add_contents);
    }

    /// `frame_text` with the frame's input events (pointer moves and clicks).
    fn frame_text_with_events(
        ctx: &egui::Context,
        size: egui::Vec2,
        events: Vec<egui::Event>,
        add_contents: impl FnMut(&egui::Context),
    ) -> Vec<(String, egui::Rect)> {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size)),
            events,
            ..Default::default()
        };
        let output = ctx.run(input, add_contents);
        output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Text(text) => Some((
                    text.galley.text().to_string(),
                    text.galley.rect.translate(text.pos.to_vec2()),
                )),
                _ => None,
            })
            .collect()
    }

    /// The default window size (`main.rs`).
    const WINDOW: egui::Vec2 = egui::vec2(1280.0, 800.0);

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
            "sf_settings",
            "sf_view",
            "pdw_settings",
            "pdw_view",
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
        assert_eq!(restored.sf_settings, SfSettings::default());
        assert_eq!(restored.sf_view, SfView::Magnetoresistance);
        assert_eq!(restored.pdw_settings, PdwSettings::default());
        assert_eq!(restored.pdw_view, PdwView::default());
    }

    #[test]
    fn test_capture_current_view_every_tab() {
        let (mut app, ctx) = pipeline_app(64);
        // The SF stage runs only on its tab, and its default view is a plot:
        // compute it with the glue map selected.
        app.sf_view = SfView::GlueGap;
        app.active_tab = Tab::SpinFluctuation;
        app.recompute_sf(&ctx);
        let all_tabs = [
            Tab::Pattern,
            Tab::Density,
            Tab::Fourier,
            Tab::MagneticField,
            Tab::CooperSurface3D,
            Tab::Graphene,
            Tab::SpinFluctuation,
            Tab::PairDensityWave,
        ];
        // With the default (non-plot) graphene/cooper views and the SF glue
        // map, every tab in both view modes produces a capturable image.
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

        app.active_tab = Tab::PairDensityWave;
        for view in PdwView::ALL.into_iter().filter(PdwView::is_plot) {
            app.pdw_view = view;
            for vm in [ViewMode::Flat2D, ViewMode::Surface3D] {
                app.view_mode = vm;
                assert!(
                    app.capture_current_view().is_none(),
                    "PDW {view:?}/{vm:?} should not capture"
                );
            }
        }

        app.active_tab = Tab::SpinFluctuation;
        for view in SfView::ALL.into_iter().filter(SfView::is_plot) {
            app.sf_view = view;
            for vm in [ViewMode::Flat2D, ViewMode::Surface3D] {
                app.view_mode = vm;
                assert!(
                    app.capture_current_view().is_none(),
                    "SF {view:?}/{vm:?} should not capture"
                );
            }
        }
    }

    #[test]
    fn test_capture_2d_puts_row_zero_at_the_bottom() {
        // Data row 0 holds y_min, which the viewport's flipped uv paints at
        // the bottom (under the axes' y_range.0 label); the app's own 2D
        // screenshot matches, so its first pixel row is the colormapped last
        // data row. A twist makes the pattern y-asymmetric (the untwisted
        // default is mirror-symmetric to 1e-14).
        let (mut app, ctx) = pipeline_app(64);
        app.twist_angle = 3.0;
        app.recompute(&ctx);
        app.active_tab = Tab::Pattern;
        app.view_mode = ViewMode::Flat2D;
        let m = app.moire_result.as_ref().unwrap();
        let n = m.resolution;
        let cmap = app.cmap(moire_core::colormap::viridis);
        let row = |i: usize| -> Vec<egui::Color32> {
            m.pattern[i * n..(i + 1) * n]
                .iter()
                .map(|&v| {
                    let [r, g, b, a] = cmap(v);
                    egui::Color32::from_rgba_premultiplied(r, g, b, a)
                })
                .collect()
        };
        assert_ne!(row(0), row(n - 1), "the test needs a y-asymmetric pattern");
        let image = app.capture_current_view().expect("pattern capture");
        assert_eq!(image.size, [n, n]);
        assert_eq!(image.pixels[..n], row(n - 1)[..]);
        assert_eq!(image.pixels[n * (n - 1)..], row(0)[..]);
    }

    #[test]
    fn test_rerender_surface_all_tabs() {
        let (mut app, ctx) = pipeline_app(64);
        app.view_mode = ViewMode::Surface3D;
        app.sf_view = SfView::GlueGap;
        app.active_tab = Tab::SpinFluctuation;
        app.recompute_sf(&ctx);
        for tab in [
            Tab::Pattern,
            Tab::Density,
            Tab::Fourier,
            Tab::MagneticField,
            Tab::CooperSurface3D,
            Tab::Graphene,
            Tab::SpinFluctuation,
            Tab::PairDensityWave,
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

    #[test]
    fn test_recompute_pdw_populates() {
        let (app, _ctx) = pipeline_app(64);
        let rt = app.pdw_result.as_ref().expect("pdw result");
        assert_eq!(rt.fields.pair_amplitude.len(), 64 * 64);
        assert_eq!(rt.fields.charge_density.len(), 64 * 64);
        assert_eq!(rt.t_sweep.values.len(), PDW_SWEEP_POINTS);
        assert_eq!(rt.b_sweep.intensity_q[2].len(), PDW_SWEEP_POINTS);
        assert_eq!(rt.peaks.len(), 8);
        // UTe2 at 300 mK, zero field: PDW and uniform SC coexist.
        assert_eq!(rt.state.regime, PdwRegime::Coexistence);
        assert!(app.pdw_texture.is_some());
    }

    #[test]
    fn test_pdw_view_scalar_all_views() {
        let (mut app, ctx) = pipeline_app(32);
        for view in PdwView::ALL {
            app.pdw_view = view;
            app.recompute_pdw(&ctx);
            if view.is_plot() {
                assert!(app.pdw_view_scalar().is_none(), "{view:?} is a plot");
                continue;
            }
            let (data, _cmap) = app.pdw_view_scalar().expect("scalar for view");
            assert_eq!(data.len(), 32 * 32, "wrong length for {view:?}");
            for &v in &data {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "value {v} out of [0,1] for {view:?}"
                );
            }
        }
    }

    #[test]
    fn test_pdw_charge_map_is_symmetric_about_midpoint() {
        // rho(r) is signed; zero must stay at the colormap midpoint.
        let (mut app, ctx) = pipeline_app(32);
        app.pdw_view = PdwView::ChargeDensity;
        app.recompute_pdw(&ctx);
        let rho = &app.pdw_result.as_ref().unwrap().fields.charge_density;
        let (data, _cmap) = app.pdw_view_scalar().unwrap();
        let max_abs = rho.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (&r, &d) in rho.iter().zip(&data) {
            assert!((d - (0.5 + 0.5 * r / max_abs)).abs() < 1e-12);
        }
    }

    #[test]
    fn test_pdw_moire_locked_uses_shared_period() {
        // FeTe / Sb2Te3 at a 1 deg twist locks to the shared two-lattice
        // period — the parity anchor also pinned in tests/test_pdw.py — not
        // to the moire stage's own min(mismatch, twist) estimate.
        let (mut app, ctx) = pipeline_app(32);
        app.twist_angle = 1.0;
        app.pdw_settings.geometry = PdwGeometry::MoireLocked;
        app.pdw_settings.tpdw_ratio = 1.0;
        app.recompute_pdw(&ctx);
        let rt = app.pdw_result.as_ref().expect("pdw result");
        let period = 36.23263757853417;
        for [kx, ky] in rt.config.p_vectors {
            let k = (kx * kx + ky * ky).sqrt();
            assert!((k - 2.0 * std::f64::consts::PI / period).abs() < 1e-12);
        }
        assert_eq!(rt.config.tc, moire_core::pdw::TC_FETE_FILM);
        assert_eq!(rt.config.t_pdw, rt.config.tc);
        assert!(rt.system.contains("L_m = 36.2"));
    }

    #[test]
    fn test_pdw_field_off_axis_has_no_marker() {
        let (mut app, ctx) = pipeline_app(32);
        app.pdw_settings.field = [0.0, 5.0, 0.0];
        app.pdw_settings.sweep_axis = PdwSweepAxis::Z;
        app.recompute_pdw(&ctx);
        assert_eq!(
            app.pdw_result.as_ref().unwrap().field_along_axis(),
            (0.0, false)
        );
        app.pdw_settings.field = [0.0, 0.0, 6.0];
        app.recompute_pdw(&ctx);
        let (along, on_axis) = app.pdw_result.as_ref().unwrap().field_along_axis();
        assert!(on_axis && (along - 6.0).abs() < 1e-12);
        // Along a zero field the sweep falls back to z, and says so.
        assert_eq!(
            PdwSweepAxis::AlongField.phrase([0.0; 3]),
            PdwSweepAxis::Z.label()
        );
        assert_eq!(
            PdwSweepAxis::AlongField.phrase([1.0, 0.0, 0.0]),
            "the applied B"
        );
    }

    #[test]
    fn test_pdw_tpdw_ratio_opens_cheshire_window() {
        let (mut app, ctx) = pipeline_app(32);
        app.pdw_settings.temperature = 3.0;
        app.recompute_pdw(&ctx);
        assert_eq!(
            app.pdw_result.as_ref().unwrap().state.regime,
            PdwRegime::PdwOnly
        );
        // With T_PDW = Tc the PDW dies with the uniform condensate.
        app.pdw_settings.tpdw_ratio = 1.0;
        app.recompute_pdw(&ctx);
        assert_eq!(
            app.pdw_result.as_ref().unwrap().state.regime,
            PdwRegime::Normal
        );
    }

    #[test]
    fn test_pdw_regime_texture_is_themed() {
        let (mut app, ctx) = pipeline_app(32);
        app.pdw_view = PdwView::RegimeMap;
        app.recompute_pdw(&ctx);
        let tex = app.pdw_texture.as_ref().expect("regime texture");
        assert_eq!(tex.size(), [PDW_REGIME_N, PDW_REGIME_N]);
        assert_eq!(app.pdw_texture_dark, Some(app.dark_mode));
        let dark = app.pdw_regime_image().unwrap();
        app.dark_mode = !app.dark_mode;
        let light = app.pdw_regime_image().unwrap();
        assert_ne!(dark.pixels, light.pixels);
    }

    #[test]
    fn test_pdw_bad_moire_reports_error() {
        // Graphene on graphene at zero twist has no moire, so the moire-locked
        // PDW cannot be built: the stage clears its caches, keeps the error for
        // the viewport, and records the theme so the regime-map re-theme check
        // does not retry the failing computation every frame.
        let (mut app, ctx) = pipeline_app(32);
        let graphene_sub = materials::substrates()
            .iter()
            .position(|m| m.formula == "Graphene")
            .expect("graphene substrate");
        let graphene_over = materials::overlayers()
            .iter()
            .position(|m| m.formula == "Graphene")
            .expect("graphene overlayer");
        app.substrate_idx = graphene_sub;
        app.overlayer_idx = graphene_over;
        app.twist_angle = 0.0;
        app.pdw_settings.geometry = PdwGeometry::MoireLocked;
        app.pdw_view = PdwView::RegimeMap;
        app.recompute_pdw(&ctx);
        assert!(app.pdw_result.is_none());
        assert!(app.pdw_texture.is_none());
        assert!(app.pdw_error.as_deref().unwrap().contains("no moire"));
        assert_eq!(app.pdw_texture_dark, Some(app.dark_mode));
        // A valid geometry clears the error again.
        app.pdw_settings.geometry = PdwGeometry::Ute2Surface;
        app.recompute_pdw(&ctx);
        assert!(app.pdw_error.is_none() && app.pdw_result.is_some());
    }

    /// Relative closeness for the SF wire-through anchors.
    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn test_recompute_sf_populates() {
        let (app, _ctx) = sf_app(64);
        assert!(app.sf_error.is_none());
        let rt = app.sf_result.as_ref().expect("sf result");
        let tr = &rt.transport;
        // MR family 4 x 71 (0..=35 T in 0.5 T), 19 doping rows, both sweeps.
        assert_eq!(tr.mr.temperatures, sf::SF_MR_TEMPERATURES.to_vec());
        assert_eq!(tr.mr.mr.len(), 4);
        assert!(tr.mr.mr.iter().all(|row| row.len() == 71));
        assert_eq!(tr.quadrature.len(), 4);
        assert_eq!(tr.doping.len(), 19);
        assert_eq!(tr.rho_t.len(), SF_RHO_T_POINTS);
        assert_eq!(tr.hall_cvc.len(), SF_HALL_T_POINTS);
        assert_eq!(rt.current.mr[0].len(), 71);
        assert_eq!(rt.gamma.len(), sf::SF_N_PHI);
        // The glue map sits on the moire grid; the imprint has 80 Tc* points.
        let glue = rt.glue.as_ref().expect("glue map");
        assert_eq!(glue.gap.len(), 64 * 64);
        assert_eq!(
            rt.glue_resolution,
            app.moire_result.as_ref().unwrap().resolution
        );
        let imprint = rt.imprint.as_ref().expect("imprint curves");
        for curve in [
            &imprint.tc_targets,
            &imprint.dressed,
            &imprint.local,
            &imprint.linear,
        ] {
            assert_eq!(curve.len(), SF_IMPRINT_POINTS);
        }
        // Clean coherence by default, so no Tc(r) = 0 shading.
        assert_eq!(app.sf_settings.coherence, SfCoherence::Clean);
        assert_eq!(rt.glue_config.unwrap().coherence, SfCoherence::Clean);
        assert!(imprint.fragment_below.is_none());
        // The default view is a plot, so there is no texture.
        assert_eq!(app.sf_view, SfView::Magnetoresistance);
        assert!(app.sf_texture.is_none());

        // The readouts carry the core's parity anchors (SF-amendments §9.3 v2)
        // at the defaults: FeTe / Sb2Te3 at 0 deg, x = 0.159, 40 K.
        let r = &rt.readouts;
        assert!(close(r.rho0, 24.49517992890017, 1e-12));
        assert!(close(r.mr_20t, 0.022350310369975812, 1e-11));
        assert!(close(r.slope_33t, 0.05026085857440421, 1e-9));
        assert!(close(r.rh_plain, 6.131175356207996e-10, 1e-12));
        assert!(close(r.rh_cvc, 5.092606357247605e-10, 1e-12));
        assert!(close(r.rh_ratio_to_case_a, 0.9592004336484117, 1e-12));
        assert!(close(r.rh_drude, 7.403312411077447e-10, 1e-12));
        assert!(close(r.crossing_deg, 27.196569908994878, 1e-12));
        assert!(close(r.lambda_tr, 0.41758273992325223, 1e-12));
        assert!(close(r.lambda_rho, 0.4352438971760371, 1e-9));
        assert!(close(r.lambda_bar, 0.6932743989110509, 1e-12));
        assert!(close(r.sensitivity, 0.700404169756287, 1e-12));
        assert!(close(r.x_equiv.unwrap(), 0.1435584333754864, 1e-12));
        assert!(close(r.x_threshold.unwrap(), 0.17052577242235883, 1e-12));
        assert_eq!(r.moire_period, Some(sf::SF_GLUE_MOIRE_PERIOD));
        assert!(close(
            r.coherence_factor.unwrap(),
            0.5797428855287498,
            1e-14
        ));
        assert!(close(
            r.glue_rel_dressed.unwrap(),
            0.04081885749393931,
            1e-12
        ));
        assert!(close(r.glue_rel_local, 0.0704085526753999, 1e-12));
        assert!(close(r.glue_rel_linear, 0.07004041697562871, 1e-12));
        assert_eq!((r.delta_bar, r.coherence_length), (3.09, 20.0));
        // xi / L_m = 0.545 > 1/2: the "local map unreliable" flag is on.
        assert!(r.xi_over_lm.unwrap() > 0.5);
        assert_eq!(r.normal_fraction, Some(0.0));
        // Anti-Kohler: MR(20 T) grows from 20 K to 50 K, and faster than
        // Kohler's rule allows (the ratio at equal B/ρ₀ is above 1).
        assert!(r.mr_50k_20t > r.mr_20k_20t);
        assert!(r.kohler_ratio > 1.0);
    }

    #[test]
    fn test_sf_kohler_ratio_at_equal_b_over_rho0() {
        // κ = MR(50 K, 20 T) / MR(20 K, 20 T · ρ₀(20 K)/ρ₀(50 K)), the web
        // `_kohler_ratio`: 1 under Kohler's rule, which a uniform or a
        // T-independent Γ obeys exactly; above 1 anti-Kohler; below 1 a fall
        // faster than the rule predicts. The card line is the web's, with
        // "->" for its arrow.
        let (mut app, ctx) = sf_app(32);
        let preset = |name: &str| {
            crate::ui::sf_panel::presets()
                .into_iter()
                .find(|p| p.name == name)
                .unwrap_or_else(|| panic!("preset {name:?}"))
                .settings
        };
        let kohler_line = |lines: &[String]| {
            lines
                .iter()
                .find(|l| l.contains("Kohler check:"))
                .map(|l| l.trim().to_string())
                .expect("the Kohler line")
        };

        // The paper's form (the defaults): anti-Kohler.
        let lines = sf_info_lines(&mut app, &ctx);
        let r = &app.sf_result.as_ref().unwrap().readouts;
        assert!(
            close(r.kohler_ratio, 2.0553114919661515, 1e-10),
            "{}",
            r.kohler_ratio
        );
        assert_eq!(
            kohler_line(&lines),
            "Kohler check: MR(20 T) = 1.950 % at 20 K -> 2.471 % at 50 K; at equal B/ρ₀, \
             MR(50 K)/MR(20 K) = 2.055: grows (anti-Kohler)"
        );

        // Isotropic Γ(T) without hot spots: MR depends on B/ρ₀ alone, although
        // MR(20 T) falls fourfold between 20 K and 50 K. The web gets exactly
        // 1.0; here the two Chambers solves round differently, and MR ≈ 1e-3
        // turns that 1e-16 noise in ρ into 1e-13 in κ.
        app.sf_settings = preset("Kohler check: isotropic Γ, no hot spots");
        let lines = sf_info_lines(&mut app, &ctx);
        let r = &app.sf_result.as_ref().unwrap().readouts;
        assert!(r.mr_50k_20t < r.mr_20k_20t);
        assert!(close(r.kohler_ratio, 1.0, 1e-11), "{}", r.kohler_ratio);
        assert_eq!(
            kohler_line(&lines),
            "Kohler check: MR(20 T) = 0.096 % at 20 K -> 0.025 % at 50 K; at equal B/ρ₀, \
             MR(50 K)/MR(20 K) = 1.000: collapses on B/ρ₀ (Kohler)"
        );

        // No inelastic term: Γ does not depend on T, so the two MRs are equal
        // (the old fixed-field rule printed "falls" beside them).
        app.sf_settings = SfSettings {
            inelastic: SfInelasticForm::None,
            ..SfSettings::default()
        };
        let lines = sf_info_lines(&mut app, &ctx);
        let r = &app.sf_result.as_ref().unwrap().readouts;
        assert_eq!(r.mr_50k_20t, r.mr_20k_20t);
        assert_eq!(r.kohler_ratio, 1.0);
        assert!(kohler_line(&lines).ends_with("= 1.000: collapses on B/ρ₀ (Kohler)"));

        // Fig. 9c: isotropic Γ(T) with strong hot spots. MR(20 T) falls, but by
        // more than the rule allows (the old rule called this "Kohler-like").
        app.sf_settings = preset("Ablation: isotropic + strong hot spots (Fig. 9c)");
        let lines = sf_info_lines(&mut app, &ctx);
        let r = &app.sf_result.as_ref().unwrap().readouts;
        assert!(r.mr_50k_20t < r.mr_20k_20t);
        assert!(
            close(r.kohler_ratio, 0.5960666742888033, 1e-10),
            "{}",
            r.kohler_ratio
        );
        assert_eq!(
            kohler_line(&lines),
            "Kohler check: MR(20 T) = 2.107 % at 20 K -> 0.438 % at 50 K; at equal B/ρ₀, \
             MR(50 K)/MR(20 K) = 0.596: falls faster than Kohler (inverted)"
        );
    }

    #[test]
    fn test_sf_fixed_cvc_note_survives_the_open_scattering_section() {
        // egui re-snaps a slider's value to its step on every draw, with binary
        // noise: 31.4 becomes 31.400000000000002 when its slider is merely
        // shown. `sf_config` rounds it back, so opening "Scattering
        // parameters" keeps the config at the paper's settings, the transport
        // cached and the fixed CVC note on the card.
        const FIXED: &str = "In this form CVCs lower R_H";
        let (mut app, ctx) = sf_app(32);
        assert_eq!(app.sf_config().unwrap().config, SfConfig::default());
        app.sf_settings.hotspot_peak = 31.400000000000002;
        assert_eq!(app.sf_config().unwrap().config, SfConfig::default());
        assert_eq!(snap_to_step(31.400000000000002, 0.1), 31.4);
        assert_eq!(snap_to_step(0.011200000000000002, 0.0002), 0.0112);
        assert_eq!(snap_to_step(27.5, 0.5), 27.5);
        app.sf_settings = SfSettings::default();

        // Tall enough for the whole sidebar (the SF panel with its open
        // section, then the info panel) to paint.
        let window = egui::vec2(1280.0, 5600.0);
        let transport = Arc::clone(&app.sf_result.as_ref().unwrap().transport);
        click_text(&ctx, window, "Scattering parameters (Table 2)", |ctx| {
            app.run_compute_stages(ctx);
            app.show_panels(ctx);
        });
        let mut text = Vec::new();
        for _ in 0..3 {
            text = frame_text(&ctx, window, |ctx| {
                app.run_compute_stages(ctx);
                app.show_panels(ctx);
            });
        }
        let has = |needle: &str| text.iter().any(|(t, _)| t.contains(needle));
        assert!(has("Hot-spot height (1/ps):"), "the section did not open");
        eprintln!(
            "hotspot_peak after its slider was drawn: {:?}",
            app.sf_settings.hotspot_peak
        );
        let rt = app.sf_result.as_ref().unwrap();
        assert_eq!(rt.transport.config, SfConfig::default());
        assert!(
            Arc::ptr_eq(&transport, &rt.transport),
            "the transport was rebuilt"
        );
        assert!(has(FIXED), "the fixed CVC note disappeared");
    }

    #[test]
    fn test_sf_view_scalar_glue_only() {
        let coolwarm = moire_core::colormap::coolwarm as fn(f64) -> [u8; 4];
        let plasma = moire_core::colormap::plasma as fn(f64) -> [u8; 4];
        let (mut app, ctx) = sf_app(32);
        app.sf_view = SfView::GlueGap;
        app.recompute_sf(&ctx);
        for view in SfView::ALL {
            app.sf_view = view;
            if view.is_plot() {
                assert!(app.sf_view_scalar().is_none(), "{view:?} is a plot");
                continue;
            }
            let (data, cmap) = app.sf_view_scalar().expect("glue map scalar");
            assert_eq!(data.len(), 32 * 32);
            assert!(data.iter().all(|v| (0.0..=1.0).contains(v)));
            assert_eq!(cmap as usize, coolwarm as usize);
        }

        // eta = 0 gives a uniform map at mid-colormap, not a blank texture.
        app.sf_view = SfView::GlueGap;
        app.sf_settings.eta = 0.0;
        app.recompute_sf(&ctx);
        let (data, _) = app.sf_view_scalar().expect("uniform glue map");
        assert!(data.iter().all(|&v| v == 0.5));
        assert!(app.sf_texture.is_some());

        // The global colormap override applies, as for every texture view.
        app.colormap_override = Some(ColormapName::Plasma);
        let (_, cmap) = app.sf_view_scalar().unwrap();
        assert_eq!(cmap as usize, plasma as usize);
    }

    #[test]
    fn test_sf_transport_cached_across_glue_changes() {
        let (mut app, ctx) = sf_app(32);
        let bundle = Arc::clone(&app.sf_result.as_ref().unwrap().transport);

        // Glue settings leave the (config, field_max) key alone.
        app.sf_settings.eta = 0.25;
        app.sf_settings.tc_target = 8.0;
        app.sf_settings.coherence = SfCoherence::Dirty;
        app.recompute_sf(&ctx);
        assert!(Arc::ptr_eq(
            &bundle,
            &app.sf_result.as_ref().unwrap().transport
        ));

        // So does T, but the current-T curve and the readouts are rebuilt.
        let before = app.sf_result.as_ref().unwrap().readouts;
        app.sf_settings.temperature = 25.0;
        app.recompute_sf(&ctx);
        let rt = app.sf_result.as_ref().unwrap();
        assert!(Arc::ptr_eq(&bundle, &rt.transport));
        let config = rt.transport.config;
        let direct = sf::magnetoresistance(&config, &[25.0], 35.0, sf::SF_FIELD_STEP).unwrap();
        assert_eq!(rt.current, direct);
        let r = sf::resistivity_tensor(&config, 25.0, &[0.0, 20.0]).unwrap();
        assert_eq!(rt.readouts.mr_20t, r[1][0] / r[0][0] - 1.0);
        assert_ne!(rt.readouts.rho0, before.rho0);

        // A doping change rebuilds the bundle, and so does field_max.
        app.sf_settings.doping = 0.116;
        app.recompute_sf(&ctx);
        let rebuilt = Arc::clone(&app.sf_result.as_ref().unwrap().transport);
        assert!(!Arc::ptr_eq(&bundle, &rebuilt));
        assert_eq!(rebuilt.config.doping, 0.116);
        app.sf_settings.field_max = 20.0;
        app.recompute_sf(&ctx);
        let rt = app.sf_result.as_ref().unwrap();
        assert!(!Arc::ptr_eq(&rebuilt, &rt.transport));
        assert_eq!(rt.transport.mr.fields.len(), 41);
    }

    #[test]
    fn test_sf_settings_snap_and_clamp() {
        let mut app = MoireApp::default();
        app.sf_settings = SfSettings {
            doping: 0.15,
            temperature: 1.0,
            field_max: 35.74,
            gamma_imp: 0.0,
            alpha1: 5.0,
            alpha2: -1.0,
            hotspot_peak: 500.0,
            hotspot_width: 1.0,
            nu: 13,
            phi_hs_deg: 50.0,
            cvc_epsilon: 2.0,
            eta: 0.9,
            tc_target: 40.0,
            ..SfSettings::default()
        };
        let inputs = app.sf_config().expect("finite settings are clamped");
        let c = &inputs.config;
        assert_eq!(c.doping, 0.149);
        assert_eq!(c.nu, 12);
        assert_eq!(inputs.glue_free.tc_target, 25.0);
        assert_eq!(inputs.glue.unwrap().tc_target, 25.0);
        assert_eq!((inputs.temperature, inputs.field_max), (2.0, 35.5));
        assert_eq!((c.gamma_imp, c.alpha1, c.alpha2), (0.5, 3.0, 0.0));
        assert_eq!((c.hotspot_peak, c.hotspot_width), (200.0, 0.3));
        assert_eq!(
            (c.phi_hs_deg, c.cvc_epsilon, inputs.glue_free.eta),
            (40.0, 1.0, 0.5)
        );
        // The glue gap scale comes from the core DELTA_AVG (I7), not the
        // sidebar gaps.
        app.density_config.delta_1 = 9.0;
        assert_eq!(app.sf_config().unwrap().glue_free.delta_avg, 3.09);

        // ν: the nearest even integer, ties down, inside [2, 24].
        for (nu, want) in [(0, 2), (3, 2), (12, 12), (15, 14), (25, 24), (99, 24)] {
            app.sf_settings.nu = nu;
            assert_eq!(app.sf_config().unwrap().config.nu, want, "nu = {nu}");
        }
        // field_max: inside [5, 65], snapped down to the 0.5 T step.
        for (field_max, want) in [(1.0, 5.0), (35.0, 35.0), (64.9, 64.5), (99.0, 65.0)] {
            app.sf_settings.field_max = field_max;
            assert_eq!(app.sf_config().unwrap().field_max, want);
        }

        // Settings saved before the coherence model existed load as clean...
        let local = SfSettings {
            coherence: SfCoherence::Local,
            ..SfSettings::default()
        };
        let mut value = serde_json::to_value(local).expect("serialize");
        let obj = value.as_object_mut().expect("object");
        assert!(obj.remove("coherence").is_some());
        let restored: SfSettings = serde_json::from_value(value).expect("deserialize");
        assert_eq!(restored, SfSettings::default());
        // ... and an empty blob gives the web base preset.
        let empty: SfSettings = serde_json::from_str("{}").expect("deserialize");
        assert_eq!(empty, SfSettings::default());
    }

    #[test]
    fn test_sf_error_reported_and_cleared() {
        // A non-finite setting (a corrupted persisted state) fails `sf_config`:
        // the stage clears its caches and keeps the error for the viewport.
        let (mut app, ctx) = sf_app(32);
        app.sf_view = SfView::GlueGap;
        app.recompute_sf(&ctx);
        assert!(app.sf_texture.is_some());
        app.sf_settings.gamma_imp = f64::NAN;
        app.recompute_sf(&ctx);
        assert!(app.sf_result.is_none());
        assert!(app.sf_texture.is_none());
        assert!(app.sf_error.as_deref().unwrap().contains("gamma_imp"));
        // A finite value clears the error again.
        app.sf_settings.gamma_imp = 27.0;
        app.recompute_sf(&ctx);
        assert!(app.sf_error.is_none());
        assert!(app.sf_result.is_some() && app.sf_texture.is_some());
    }

    #[test]
    fn test_sf_readouts_independent_of_field_max() {
        // I4: the readouts are direct evaluations, so a 5 T field grid (11
        // points) changes the plots but not one readout.
        let (mut app, ctx) = sf_app(32);
        let at_35 = app.sf_result.as_ref().unwrap().readouts;
        app.sf_settings.field_max = 5.0;
        app.recompute_sf(&ctx);
        assert!(app.sf_error.is_none());
        let rt = app.sf_result.as_ref().unwrap();
        assert_eq!(rt.transport.field_max, 5.0);
        assert!(rt.transport.mr.mr.iter().all(|row| row.len() == 11));
        assert_eq!(rt.current.fields.len(), 11);
        assert_eq!(rt.readouts, at_35);
    }

    #[test]
    fn test_sf_recompute_only_on_active_tab() {
        // I2: the SF stage runs only while its tab is open.
        let mut app = MoireApp::default();
        app.resolution = 32;
        let ctx = egui::Context::default();
        app.run_compute_stages(&ctx);
        assert!(app.moire_result.is_some() && app.pdw_result.is_some());
        assert!(app.needs_sf_recompute);
        assert!(app.sf_result.is_none());

        app.active_tab = Tab::SpinFluctuation;
        app.run_compute_stages(&ctx);
        assert!(!app.needs_sf_recompute);
        assert!(app.sf_result.is_some());
        assert!(!crate::ui::sf_panel::sf_stale(&app));

        // A main-pipeline change on another tab marks it dirty again and keeps
        // the last values for the readouts.
        app.active_tab = Tab::Pattern;
        app.needs_recompute = true;
        app.run_compute_stages(&ctx);
        assert!(app.needs_sf_recompute);
        assert!(app.sf_result.is_some());
        assert!(crate::ui::sf_panel::sf_stale(&app));
    }

    #[test]
    fn test_sf_no_moire_glue_message() {
        // Graphene on graphene at zero twist has no moire (L_m infinite). The
        // transport still computes; the glue map and imprint carry the PDW
        // wording instead of failing the stage.
        let (mut app, ctx) = pipeline_app(32);
        app.substrate_idx = material_index(&materials::substrates(), "Graphene");
        app.overlayer_idx = material_index(&materials::overlayers(), "Graphene");
        app.twist_angle = 0.0;
        app.recompute(&ctx);
        app.active_tab = Tab::SpinFluctuation;
        app.sf_view = SfView::GlueGap;
        app.recompute_sf(&ctx);
        assert!(app.sf_error.is_none());
        let rt = app
            .sf_result
            .as_ref()
            .expect("the transport still computes");
        assert_eq!(rt.transport.mr.mr.len(), 4);
        assert!(rt.glue.is_none() && rt.imprint.is_none() && rt.glue_config.is_none());
        assert_eq!(
            rt.glue_message.as_deref(),
            Some(
                "Graphene/Graphene has no moire at this twist; pick different layers or a \
                 nonzero twist"
            )
        );
        assert!(app.sf_texture.is_none());
        assert!(app.capture_current_view().is_none());

        // The six L_m-dependent readouts are None; the rest are filled.
        let r = &rt.readouts;
        let lm_dependent = [
            r.moire_period,
            r.xi_over_lm,
            r.coherence_factor,
            r.glue_rel_dressed,
            r.local_normal_fraction,
            r.normal_fraction,
        ];
        assert_eq!(lm_dependent, [None; 6]);
        for value in [
            r.lambda_bar,
            r.sensitivity,
            r.delta_bar,
            r.coherence_length,
            r.glue_rel_local,
            r.glue_rel_linear,
            r.lambda_tr,
            r.lambda_rho,
            r.rho0,
            r.mr_20t,
            r.rh_plain,
            r.rh_cvc,
        ] {
            assert!(value.is_finite());
        }
        assert!(r.x_equiv.is_some());

        // The info panel keeps only the L_m-free glue lines.
        let text = frame_text(&ctx, egui::vec2(400.0, 2000.0), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui::info_panel::show_info_panel(ui, &app);
            });
        });
        let shown = |needle: &str| text.iter().any(|(t, _)| t.contains(needle));
        assert!(shown("no moire at this twist"));
        assert!(shown("<λ> = 0.693"));
        assert!(shown("δΔ/<Δ>: local 0.070"));
        assert!(!shown("ξ/L_m") && !shown("dressed"));
        // The viewport explains instead of "Computing...".
        let text = frame_text(&ctx, WINDOW, |ctx| app.show_panels(ctx));
        assert!(text
            .iter()
            .any(|(t, _)| t.contains("Graphene/Graphene has no moire at this twist")));
    }

    #[test]
    fn test_sf_every_preset_and_view_renders() {
        // Each of the 11 web presets computes, and every view draws a headless
        // frame of the real layout with neither an error nor a placeholder.
        let (mut app, ctx) = pipeline_app(32);
        app.active_tab = Tab::SpinFluctuation;
        let presets = crate::ui::sf_panel::presets();
        let mut names: Vec<&str> = presets.iter().map(|p| p.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), 11);
        for preset in presets {
            app.sf_settings = preset.settings;
            app.sf_view = preset.view;
            app.recompute_sf(&ctx);
            assert!(
                app.sf_error.is_none(),
                "{}: {:?}",
                preset.name,
                app.sf_error
            );
            for view in SfView::ALL {
                app.sf_view = view;
                if !view.is_plot() {
                    app.recompute_sf(&ctx);
                }
                let text = frame_text(&ctx, WINDOW, |ctx| app.show_panels(ctx));
                let placeholder = text
                    .iter()
                    .find(|(t, _)| t.contains("unavailable") || t.contains("Computing"));
                assert!(
                    placeholder.is_none(),
                    "{} / {view:?}: {placeholder:?}",
                    preset.name
                );
            }
        }
    }

    #[test]
    fn test_tab_bar_fits_at_1280() {
        // I10: the tab bar, now with "Spin Fluctuations", fits the default
        // 1280 px window beside the sidebar on one row. Its last item is the
        // "3D" toggle.
        let mut app = MoireApp::default();
        let ctx = egui::Context::default();
        app.run_compute_stages(&ctx);
        // The side panel settles its width after the first frame.
        let mut text = Vec::new();
        for _ in 0..3 {
            text = frame_text(&ctx, WINDOW, |ctx| app.show_panels(ctx));
        }
        let rect_of = |label: &str| {
            text.iter()
                .find(|(t, _)| t == label)
                .map(|(_, r)| *r)
                .unwrap_or_else(|| panic!("{label:?} was not painted"))
        };
        let first = rect_of("Moire Pattern");
        let sf_tab = rect_of("Spin Fluctuations");
        let toggle = rect_of("3D");
        for rect in [sf_tab, toggle] {
            assert!(
                (rect.center().y - first.center().y).abs() < 1.0,
                "the tab bar wrapped"
            );
        }
        let sidebar = egui::containers::panel::PanelState::load(&ctx, egui::Id::new("controls"))
            .expect("side panel state")
            .rect
            .width();
        // The central panel's right content edge (8 px frame margin).
        let right_edge = WINDOW.x - 8.0;
        let slack = right_edge - toggle.max.x;
        // The user may widen the sidebar to its 350 px maximum.
        let slack_at_max = slack - (350.0 - sidebar);
        eprintln!(
            "tab bar: sidebar {sidebar:.0} px, bar {:.0}..{:.0} px, slack {slack:.0} px \
             ({slack_at_max:.0} px with the sidebar at 350 px)",
            first.min.x, toggle.max.x
        );
        assert!(slack >= 0.0, "the tab bar overflows by {:.0} px", -slack);
        assert!(
            slack_at_max >= 0.0,
            "a 350 px sidebar would push the tab bar {:.0} px past the window",
            -slack_at_max
        );
    }

    #[test]
    fn test_tab_bar_wraps_in_a_narrow_window() {
        // Narrower than about 1100 px (the window's 800 x 600 minimum, or a
        // zoomed UI) the bar wraps instead of clipping its last tabs and the
        // 2D/3D toggle, which moves as one unit: at 1085 px "2D" alone would
        // still fit on the first row.
        let mut app = MoireApp::default();
        let ctx = egui::Context::default();
        app.run_compute_stages(&ctx);
        for width in [800.0, 1085.0] {
            let window = egui::vec2(width, 600.0);
            let mut text = Vec::new();
            for _ in 0..3 {
                text = frame_text(&ctx, window, |ctx| app.show_panels(ctx));
            }
            let rect_of = |label: &str| {
                text.iter()
                    .find(|(t, _)| t == label)
                    .map(|(_, r)| *r)
                    .unwrap_or_else(|| panic!("{label:?} was not painted at {width} px"))
            };
            for label in ["Spin Fluctuations", "Pair Density Wave", "2D", "3D"] {
                let rect = rect_of(label);
                assert!(
                    rect.max.x <= window.x,
                    "{label:?} ends at {:.0} in a {width} px window",
                    rect.max.x
                );
            }
            assert!(
                rect_of("3D").min.y > rect_of("Moire Pattern").max.y,
                "the tab bar did not wrap at {width} px"
            );
            assert!(
                (rect_of("2D").center().y - rect_of("3D").center().y).abs() < 1.0,
                "the 2D/3D toggle split across rows at {width} px"
            );
        }
    }

    /// The info panel's painted lines after an SF recompute.
    fn sf_info_lines(app: &mut MoireApp, ctx: &egui::Context) -> Vec<String> {
        app.recompute_sf(ctx);
        assert!(app.sf_error.is_none(), "{:?}", app.sf_error);
        let app = &*app;
        frame_text(ctx, egui::vec2(400.0, 3000.0), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui::info_panel::show_info_panel(ui, app);
            });
        })
        .into_iter()
        .map(|(text, _)| text)
        .collect()
    }

    #[test]
    fn test_sf_info_panel_follows_the_settings() {
        // The card lines that depend on the settings: the regime blurb, one
        // "fails" on the pairing map, the Fig. 13 panel of the scattering form,
        // what the CVCs do here, and the fixed CVC note only at the paper's
        // own settings, where its numbers hold.
        const FIXED: &str = "In this form CVCs lower R_H";
        let (mut app, ctx) = sf_app(32);
        let has = |lines: &[String], needle: &str| lines.iter().any(|l| l.contains(needle));
        let ratio_line = |lines: &[String]| {
            lines
                .iter()
                .find(|l| l.contains("with CVC at T:"))
                .cloned()
                .expect("the R_H ratio line")
        };

        let lines = sf_info_lines(&mut app, &ctx);
        assert!(has(&lines, "Overdoped and superconducting"));
        assert_eq!(
            ratio_line(&lines).trim(),
            "(d)/(a) with CVC at T: 0.96 (paper Fig. 13: 4.3/8.7 = 0.49)"
        );
        assert!(has(
            &lines,
            "At T, CVCs lower R_H by 16.9 %; over 2–100 K they flatten its T-dependence."
        ));
        assert!(has(&lines, FIXED));

        // The McMillan label already ends in "— fails".
        app.sf_settings.pairing_form = SfPairingForm::McMillan;
        let lines = sf_info_lines(&mut app, &ctx);
        let pairing = lines.iter().find(|l| l.contains("Pairing map:")).unwrap();
        assert_eq!(pairing.matches("fails").count(), 1, "{pairing}");
        app.sf_settings.pairing_form = SfPairingForm::PowerLaw;

        // Antinodal with hot spots is Fig. 13c: the CVCs raise R_H and steepen
        // it, so neither the 0.49 nor the fixed note applies.
        app.sf_settings.inelastic = SfInelasticForm::Antinodal;
        let lines = sf_info_lines(&mut app, &ctx);
        assert_eq!(
            ratio_line(&lines).trim(),
            "(c)/(a) with CVC at T: 3.18 (paper Fig. 13: 10.3/8.7 = 1.18)"
        );
        assert!(has(
            &lines,
            "At T, CVCs raise R_H by 46.2 %; over 2–100 K they steepen its T-dependence."
        ));
        assert!(!has(&lines, FIXED));

        // No inelastic term with hot spots is Fig. 13b; Γ has no T-dependence.
        app.sf_settings.inelastic = SfInelasticForm::None;
        let lines = sf_info_lines(&mut app, &ctx);
        assert_eq!(
            ratio_line(&lines).trim(),
            "(b)/(a) with CVC at T: 0.93 (paper Fig. 13: 5.6/8.7 = 0.64)"
        );
        assert!(has(
            &lines,
            "At T, CVCs lower R_H by 32.3 %; R_H(T) is flat over 2–100 K."
        ));

        // Isotropic scattering has no Fig. 13 panel to compare with.
        app.sf_settings.inelastic = SfInelasticForm::Isotropic;
        let lines = sf_info_lines(&mut app, &ctx);
        assert_eq!(ratio_line(&lines).trim(), "R_H/R_H(a) with CVC at T: 0.95");
        app.sf_settings.inelastic = SfInelasticForm::AfmBoundary;

        // With ε₀ = 0 the fixed note would contradict R_H's equal values.
        app.sf_settings.cvc_epsilon = 0.0;
        let lines = sf_info_lines(&mut app, &ctx);
        assert!(has(&lines, "CVCs are off (ε₀ = 0)."));
        assert!(!has(&lines, FIXED));
        app.sf_settings.cvc_epsilon = 1.0;

        // Below x = 0.14 the blurb carries the reconstruction caveat; the
        // fixed note's x = 0.159 numbers give way to the computed line.
        app.sf_settings.doping = 0.116;
        let lines = sf_info_lines(&mut app, &ctx);
        assert!(has(&lines, "Doping regime: Fermi-surface reconstruction"));
        assert!(has(
            &lines,
            "the measured MR gains a negative low-T component this model omits"
        ));
        assert!(has(&lines, "they flatten its T-dependence") && !has(&lines, FIXED));
        app.sf_settings.doping = 0.188;
        let lines = sf_info_lines(&mut app, &ctx);
        assert!(has(&lines, "g(x) has fallen about tenfold"));
    }

    #[test]
    fn test_sf_derivative_and_imprint_labels() {
        // The derivative view draws a temperature outside the 20–50 K family
        // dashed, as the MR view and the web do.
        let (mut app, ctx) = sf_app(32);
        app.sf_view = SfView::FieldDerivative;
        for (t, shown) in [(35.0, true), (40.0, false)] {
            app.sf_settings.temperature = t;
            app.recompute_sf(&ctx);
            let text = frame_text(&ctx, WINDOW, |ctx| app.show_panels(ctx));
            let label = format!("{t:.1} K (current T)");
            assert_eq!(text.iter().any(|(s, _)| *s == label), shown, "T = {t} K");
        }

        // The imprint's comparison with 0.165 holds at one L_m (amendments
        // §4.3), so its title states the period.
        app.sf_view = SfView::Imprint;
        let text = frame_text(&ctx, WINDOW, |ctx| app.show_panels(ctx));
        let title = "Glue imprint vs operating Tc*, Sb₂Te₃/FeTe, θ = 0.0°, L_m = 36.7 Å \
                     (SPECULATIVE)";
        assert!(text.iter().any(|(s, _)| s == title));
    }

    #[test]
    fn test_about_lists_mirror_the_web_home_page() {
        // F1's About window is the desktop copy of the web home page's About
        // accordion (pages/home.py): the same Validated / Speculative split
        // (3D proximity and Majorana modes are speculative) and the same
        // source for α = 0.4 (config.py cites PRB 82, 212505).
        let ctx = egui::Context::default();
        let mut open = true;
        let mut text = Vec::new();
        for _ in 0..3 {
            text = frame_text(&ctx, WINDOW, |ctx| ui::about::show(ctx, &mut open));
        }
        let heading = |label: &str| {
            text.iter()
                .position(|(t, _)| t == label)
                .unwrap_or_else(|| panic!("{label:?} was not painted"))
        };
        let validated = heading("Validated modules");
        let speculative = heading("Speculative modules");
        assert!(validated < speculative);
        // The bullets painted in `range`, in order, begin with `names`.
        let bullets_are = |range: std::ops::Range<usize>, names: &[&str]| {
            let bullets: Vec<&str> = text[range]
                .iter()
                .filter_map(|(t, _)| t.strip_prefix("• "))
                .collect();
            assert_eq!(bullets.len(), names.len(), "{bullets:?}");
            for (bullet, name) in bullets.iter().zip(names) {
                assert!(bullet.starts_with(*name), "{bullet:?} is not {name:?}");
            }
        };
        bullets_are(
            validated..speculative,
            &["Moire patterns", "Gap modulation", "FFT analysis"],
        );
        bullets_are(
            speculative..text.len(),
            &[
                "Isotope effects",
                "Topological proximity / Majorana modes",
                "Abrikosov vortex lattice",
                "Spin-fluctuation pairing",
                "Pair-density-wave modalities",
            ],
        );
        let isotope = text
            .iter()
            .find(|(t, _)| t.starts_with("• Isotope effects"))
            .map(|(t, _)| t.as_str())
            .unwrap();
        assert!(isotope.contains(
            "α = 0.4 is the corrected iron-based isotope-effect consensus, \
             Khasanov et al., PRB 82, 212505 (2010)"
        ));
        assert!(!isotope.contains("Ba(Fe,Co)"));
    }
}
