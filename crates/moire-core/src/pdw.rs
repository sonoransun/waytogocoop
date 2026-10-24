//! Pair-density-wave (PDW) modalities and the charge orders they leave behind.
//!
//! **SPECULATIVE** throughout — a phenomenological Landau model, not a
//! microscopic theory. It implements the free energy Zhu et al. use to explain
//! the field- and temperature-sensitive charge density waves (CDWs) resolved by
//! vector-field STM on the spin-triplet superconductor UTe2 (PNAS 123,
//! e2602117123 (2026); arXiv:2603.08688). Minimising over the CDW amplitudes
//! gives three families of composite charge order:
//!
//!   - p_i: rho ~ d*_0 . d_{p_i} + d_0 . d*_{-p_i} — PDW x uniform SC, only
//!     below Tc.
//!   - q_i = 2 p_i: rho ~ d*_{-p_i} . d_{p_i} — the PDW alone, surviving
//!     between Tc and T_PDW (the "Cheshire cat's grin" regime).
//!   - h_1 = 2p_3 - 2p_2, h_2 = 2p_3 - 2p_1: rho ~ rho_{q3} rho*_{q_j}, fourth
//!     order in the PDW.
//!
//! Beyond the paper: linear Ginzburg-Landau amplitudes
//! |d|^2/|d(0)|^2 = max(0, 1 - T/T* - h) with an ellipsoidal reduced field,
//! a field-induced d-vector rotation sin(theta) = sin(theta_0) min(1, 1/h_rot)
//! for the field-induced p1/p2, leading-order CDW amplitudes, and the paper's
//! predicted composite SDW |d_0 x d_{p_i}| carried with the p-CDW coupling.
//! Field triples are (x, y, z) in the paper's vector-magnet frame
//! (x ~ c*, y = a, z ~ b* for UTe2).
//!
//! Mirrors `src/waytogocoop/computation/pdw.py` in the Python stack — changes
//! here must land in both.

use num_complex::Complex64;
use rayon::prelude::*;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants and UTe2 defaults (Zhu et al. 2026)
// ---------------------------------------------------------------------------

/// Boltzmann constant in eV/K.
const KB_EV_K: f64 = 8.617333262e-5;
/// Weak-coupling BCS ratio Delta(0) / (k_B Tc).
const BCS_GAP_RATIO: f64 = 1.764;

/// Uniform-SC transition of molten-salt-flux UTe2, in K.
pub const TC_UTE2: f64 = 2.1;
/// PDW onset, taken as the q_i CDW onset, in K.
pub const T_PDW_UTE2: f64 = 5.0;
/// Te-chain repeat in Angstrom; |q_Te| = 2 pi / a.
pub const A_TE_UTE2: f64 = 4.16;
/// Approximate uniform-SC Hc2 (x ~ c*, y = a, z ~ b*) in Tesla; b* > c* > a.
/// b* from Wu et al. (2024); a, c* scale the early-crystal 6.5 / 11 T of
/// Aoki et al. (2020) by (2.1 / 1.6)^2.
pub const HC2_UTE2: [f64; 3] = [20.0, 11.0, 34.0];
/// PDW critical fields at T = 0 (Tesla), placing H_sup(4.2 K) mid-window.
pub const H_PDW_UTE2: [f64; 3] = [14.0, 3.75, 37.5];
/// d-vector rotation onset fields (Tesla), just below where p2 appears.
pub const H_ROT_UTE2: [f64; 3] = [0.4, 0.4, 4.5];
/// d0 . d_{p_i} at zero field: p1/p2 orthogonal, p3 aligned.
pub const OVERLAP0_UTE2: [f64; 3] = [0.0, 0.0, 1.0];
/// PDW wavevectors in units of |q_Te| (mirror-consistent h2 = (0.26, +0.14)).
pub const P_VECTORS_UTE2_QTE: [[f64; 2]; 3] = [[-0.13, -0.35], [0.13, -0.35], [0.0, -0.28]];
/// Delta_p(0) / Delta_0(0) — illustrative only.
pub const PDW_RATIO_UTE2: f64 = 0.3;
/// Weak-coupling BCS gap for Tc = 2.1 K, in meV.
pub const DELTA0_UTE2: f64 = BCS_GAP_RATIO * KB_EV_K * 1.0e3 * TC_UTE2;
/// p-CDW coupling g1*Delta0*Delta_p/r1.
pub const KAPPA_P_DEFAULT: f64 = 1.0;
/// q-CDW coupling g2*Delta_p^2/r2.
pub const KAPPA_Q_DEFAULT: f64 = 1.0;
/// Harmonic coupling gamma2/r4.
pub const KAPPA_H_DEFAULT: f64 = 1.0;
/// p-harmonic feed of q, gamma1/r2 (0 is the paper's reading).
pub const KAPPA_HARM_DEFAULT: f64 = 0.0;
/// |rho_k|^2 below this counts as FFT background.
pub const VISIBILITY_FLOOR_DEFAULT: f64 = 0.01;

/// Stoichiometric FeTe film Tc (Yan et al. 2026), in K — moire-locked preset.
pub const TC_FETE_FILM: f64 = 13.5;
/// FeTe upper critical field in Tesla; mirrors Python `BC2_FETE`.
pub const BC2_FETE: f64 = 47.0;
/// Below this squared denominator the moire period is infinite; mirrors
/// Python `ZERO_THRESHOLD`.
const ZERO_THRESHOLD: f64 = 1e-12;

/// Moire-locked directions in degrees before twist: p1/p2 mirror partners,
/// p3 along -y, matching the UTe2 labelling.
const MOIRE_ANGLES_DEG: [f64; 3] = [30.0, 150.0, 270.0];

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Where the PDW wavevectors come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum PdwGeometry {
    /// UTe2 (01-1) surface, Zhu et al. 2026.
    #[default]
    Ute2Surface,
    /// Three-fold set at the heterostructure's moire period (what-if).
    MoireLocked,
}

impl PdwGeometry {
    pub const ALL: [PdwGeometry; 2] = [PdwGeometry::Ute2Surface, PdwGeometry::MoireLocked];

    pub fn label(&self) -> &'static str {
        match self {
            PdwGeometry::Ute2Surface => "UTe2 surface (Zhu 2026)",
            PdwGeometry::MoireLocked => "Moire-locked (what-if)",
        }
    }
}

/// Which orders are present at a given (T, B).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PdwRegime {
    Normal,
    /// PDW without uniform SC — only the q and h CDWs survive.
    PdwOnly,
    UniformOnly,
    Coexistence,
}

impl PdwRegime {
    /// Ordered like Python's `PDW_REGIMES`.
    pub const ALL: [PdwRegime; 4] = [
        PdwRegime::Normal,
        PdwRegime::PdwOnly,
        PdwRegime::UniformOnly,
        PdwRegime::Coexistence,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            PdwRegime::Normal => "Normal",
            PdwRegime::PdwOnly => "PDW only (Cheshire cat)",
            PdwRegime::UniformOnly => "Uniform SC only",
            PdwRegime::Coexistence => "PDW + uniform SC",
        }
    }
}

/// Parameters of the PDW Landau model; defaults are UTe2 (Zhu et al. 2026).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct PdwConfig {
    /// PDW wavevectors p_i in 1/Angstrom.
    pub p_vectors: [[f64; 2]; 3],
    /// Temperature in K.
    pub temperature: f64,
    /// Uniform-SC transition in K.
    pub tc: f64,
    /// PDW onset in K.
    pub t_pdw: f64,
    /// Applied field (x, y, z) in Tesla.
    pub field: [f64; 3],
    /// Uniform-SC critical fields at T = 0 (Tesla).
    pub hc2: [f64; 3],
    /// PDW critical fields at T = 0 (Tesla).
    pub h_pdw: [f64; 3],
    /// d-vector rotation onset fields (Tesla).
    pub h_rot: [f64; 3],
    /// Uniform gap at T = 0 in meV.
    pub delta0_mev: f64,
    /// Delta_p(0) / Delta_0(0).
    pub pdw_ratio: f64,
    /// d0 . d_{p_i} at zero field, in [0, 1].
    pub overlap0: [f64; 3],
    /// PDW sliding phases phi_i in radians.
    pub phases: [f64; 3],
    pub kappa_p: f64,
    pub kappa_q: f64,
    pub kappa_h: f64,
    pub kappa_harm: f64,
    /// |rho|^2 visibility threshold.
    pub floor: f64,
}

impl Default for PdwConfig {
    fn default() -> Self {
        Self {
            p_vectors: ute2_wavevectors(A_TE_UTE2).expect("A_TE_UTE2 is positive"),
            temperature: 0.3,
            tc: TC_UTE2,
            t_pdw: T_PDW_UTE2,
            field: [0.0; 3],
            hc2: HC2_UTE2,
            h_pdw: H_PDW_UTE2,
            h_rot: H_ROT_UTE2,
            delta0_mev: DELTA0_UTE2,
            pdw_ratio: PDW_RATIO_UTE2,
            overlap0: OVERLAP0_UTE2,
            phases: [0.0; 3],
            kappa_p: KAPPA_P_DEFAULT,
            kappa_q: KAPPA_Q_DEFAULT,
            kappa_h: KAPPA_H_DEFAULT,
            kappa_harm: KAPPA_HARM_DEFAULT,
            floor: VISIBILITY_FLOOR_DEFAULT,
        }
    }
}

impl PdwConfig {
    /// PDW model locked to a heterostructure's moire period (what-if preset).
    ///
    /// The moire CPDM is imprinted, not spontaneous, so `t_pdw = tc`: every
    /// composite order vanishes at Tc. Raising `t_pdw` above `tc` asks what an
    /// intrinsic PDW at the moire wavevector would add — a q = 2Q_m charge
    /// modulation (half the moire period) surviving above Tc. Singlet-like:
    /// d-vectors aligned, isotropic Bc2.
    pub fn moire_locked(
        moire_period: f64,
        twist_angle_deg: f64,
        delta0_mev: f64,
        pdw_ratio: f64,
    ) -> Result<Self, String> {
        let isotropic = [BC2_FETE; 3];
        Ok(Self {
            p_vectors: moire_locked_wavevectors(moire_period, twist_angle_deg)?,
            temperature: 0.3 * TC_FETE_FILM,
            tc: TC_FETE_FILM,
            t_pdw: TC_FETE_FILM,
            hc2: isotropic,
            h_pdw: isotropic,
            h_rot: isotropic,
            delta0_mev,
            pdw_ratio,
            overlap0: [1.0; 3],
            ..Self::default()
        })
    }
}

/// Mean-field PDW state and its composite orders at one (T, B).
#[derive(Debug, Clone, PartialEq)]
pub struct PdwState {
    /// |d_0(T, B)| / |d_0(0)|.
    pub delta0: f64,
    /// |d_p(T, B)| / |d_p(0)|.
    pub delta_p: f64,
    /// d0 . d_{p_i} after field rotation.
    pub overlap: [f64; 3],
    pub regime: PdwRegime,
    /// p_i CDW (PDW x uniform SC).
    pub rho_p: [Complex64; 3],
    /// q_i = 2 p_i CDW (PDW alone).
    pub rho_q: [Complex64; 3],
    /// h_1, h_2 harmonics.
    pub rho_h: [Complex64; 2],
    /// **SPECULATIVE** composite SDW at p_i.
    pub spin_p: [Complex64; 3],
}

/// Composite CDW family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdwFamily {
    P,
    Q,
    H,
}

/// One composite-order Fourier peak (the +k member of a +-k pair).
#[derive(Debug, Clone, PartialEq)]
pub struct PdwPeak {
    pub label: &'static str,
    pub family: PdwFamily,
    /// 1/Angstrom.
    pub kx: f64,
    pub ky: f64,
    /// |rho_k|.
    pub amplitude: f64,
    /// |rho_k|^2 above the visibility floor.
    pub visible: bool,
}

impl PdwPeak {
    pub fn intensity(&self) -> f64 {
        self.amplitude * self.amplitude
    }
}

/// Real-space maps of the PDW state; all grids row-major NxN.
#[derive(Debug, Clone)]
pub struct PdwFields {
    /// |d(r)| in meV.
    pub pair_amplitude: Vec<f64>,
    /// rho(r), dimensionless.
    pub charge_density: Vec<f64>,
    /// **SPECULATIVE** S_z(r), dimensionless.
    pub spin_density: Vec<f64>,
    pub resolution: usize,
    pub physical_extent: f64,
}

/// Order parameters and peak intensities |rho_k|^2 along a 1D sweep.
#[derive(Debug, Clone)]
pub struct PdwSweep {
    /// Swept parameter (K or Tesla).
    pub values: Vec<f64>,
    pub delta0: Vec<f64>,
    pub delta_p: Vec<f64>,
    pub intensity_p: [Vec<f64>; 3],
    pub intensity_q: [Vec<f64>; 3],
    pub intensity_h: [Vec<f64>; 2],
}

// ---------------------------------------------------------------------------
// Wavevectors
// ---------------------------------------------------------------------------

/// `v > 0`, false for NaN (validation must reject NaN, unlike `v <= 0.0`).
fn positive(v: f64) -> bool {
    v > 0.0
}

/// `v >= 0`, false for NaN.
fn non_negative(v: f64) -> bool {
    v >= 0.0
}

/// PDW wavevectors p_i on the UTe2 surface (1/Angstrom).
pub fn ute2_wavevectors(a_te: f64) -> Result<[[f64; 2]; 3], String> {
    if !positive(a_te) {
        return Err(format!("a_te must be positive, got {a_te}"));
    }
    let q_te = 2.0 * PI / a_te;
    Ok(P_VECTORS_UTE2_QTE.map(|[x, y]| [q_te * x, q_te * y]))
}

/// Three PDW wavevectors of magnitude 2 pi / L_m at 120 deg, rotated with
/// the overlayer twist (1/Angstrom). The three-fold set follows the
/// hexagonal overlayers; on the square FeTe substrate the moire itself is not
/// C3, so this is an assumption.
pub fn moire_locked_wavevectors(
    moire_period: f64,
    twist_angle_deg: f64,
) -> Result<[[f64; 2]; 3], String> {
    if !moire_period.is_finite() || moire_period <= 0.0 {
        return Err(format!(
            "moire_period must be positive and finite, got {moire_period}"
        ));
    }
    let q = 2.0 * PI / moire_period;
    Ok(MOIRE_ANGLES_DEG.map(|deg| {
        let angle = (deg + twist_angle_deg).to_radians();
        [q * angle.cos(), q * angle.sin()]
    }))
}

/// Two-lattice moire period L = a1 a2 / sqrt(a1^2 + a2^2 - 2 a1 a2 cos(theta)),
/// exact in both the mismatch (theta = 0) and equal-lattice twist limits and
/// infinite for identical lattices at zero twist. The moire-locked PDW uses
/// it instead of `MoireResult::moire_period` so both stacks lock to the same
/// wavevector at any twist.
pub fn moire_locked_period(
    substrate_a: f64,
    overlayer_a: f64,
    twist_angle_deg: f64,
) -> Result<f64, String> {
    if !positive(substrate_a) || !positive(overlayer_a) {
        return Err(format!(
            "lattice constants must be positive, got {substrate_a} and {overlayer_a}"
        ));
    }
    let theta = twist_angle_deg.to_radians();
    let d2 = substrate_a * substrate_a + overlayer_a * overlayer_a
        - 2.0 * substrate_a * overlayer_a * theta.cos();
    if d2 <= ZERO_THRESHOLD {
        return Ok(f64::INFINITY);
    }
    Ok(substrate_a * overlayer_a / d2.sqrt())
}

/// Delta_p(0) / Delta_0(0) of the moire-locked PDW from the two-gap CPDM:
/// half the gap split (Delta_2 - Delta_1)/2 per ±Q relative to the mean gap,
/// times the CPDM suppression exp(-xi / L_m).
pub fn moire_locked_pdw_ratio(
    moire_period: f64,
    coherence_length: f64,
    delta_1: f64,
    delta_2: f64,
) -> Result<f64, String> {
    if !positive(moire_period) || !positive(coherence_length) {
        return Err(format!(
            "moire_period and coherence_length must be positive, got {moire_period} and \
             {coherence_length}"
        ));
    }
    if !positive(delta_1 + delta_2) {
        return Err("the mean gap must be positive".to_string());
    }
    let amplitude = (delta_2 - delta_1) / 2.0;
    let average = (delta_1 + delta_2) / 2.0;
    Ok(0.5 * amplitude / average * (-coherence_length / moire_period).exp())
}

/// Harmonic wavevectors h1 = 2p3 - 2p2 and h2 = 2p3 - 2p1.
pub fn harmonic_vectors(p: &[[f64; 2]; 3]) -> [[f64; 2]; 2] {
    let h = |j: usize| [2.0 * (p[2][0] - p[j][0]), 2.0 * (p[2][1] - p[j][1])];
    [h(1), h(0)]
}

// ---------------------------------------------------------------------------
// Mean-field amplitudes and d-vector geometry
// ---------------------------------------------------------------------------

fn check_positive_triple(name: &str, values: &[f64; 3]) -> Result<(), String> {
    if values.iter().all(|&v| positive(v)) {
        Ok(())
    } else {
        Err(format!(
            "{name} must be three positive fields, got {values:?}"
        ))
    }
}

fn unit(direction: &[f64; 3]) -> Result<[f64; 3], String> {
    let norm =
        (direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2])
            .sqrt();
    if !norm.is_finite() || norm < 1e-15 {
        return Err(format!(
            "direction must be a nonzero 3-vector, got {direction:?}"
        ));
    }
    Ok(direction.map(|d| d / norm))
}

/// sqrt(sum_n (v_n / H_n)^2), summed in axis order like the NumPy twin.
fn ellipsoidal_norm(v: &[f64; 3], critical: &[f64; 3]) -> f64 {
    v.iter()
        .zip(critical)
        .map(|(b, h)| {
            let r = b / h;
            r * r
        })
        .sum::<f64>()
        .sqrt()
}

/// Ellipsoidal reduced field h = sqrt(sum_n (B_n / H_n)^2).
pub fn reduced_field(field: &[f64; 3], critical: &[f64; 3]) -> Result<f64, String> {
    check_positive_triple("critical", critical)?;
    Ok(ellipsoidal_norm(field, critical))
}

/// Normalised mean-field amplitude sqrt(max(0, 1 - T/T* - h)).
pub fn landau_amplitude(temperature: f64, t_star: f64, reduced: f64) -> Result<f64, String> {
    if !positive(t_star) {
        return Err(format!("t_star must be positive, got {t_star}"));
    }
    Ok((1.0 - temperature / t_star - reduced).max(0.0).sqrt())
}

/// Field magnitude along `direction` that kills the order at `temperature`
/// (Tesla).
pub fn critical_field_along(
    t_star: f64,
    critical: &[f64; 3],
    temperature: f64,
    direction: &[f64; 3],
) -> Result<f64, String> {
    if !positive(t_star) {
        return Err(format!("t_star must be positive, got {t_star}"));
    }
    let u = unit(direction)?;
    check_positive_triple("critical", critical)?;
    Ok((1.0 - temperature / t_star).max(0.0) / ellipsoidal_norm(&u, critical))
}

/// d-vector rotation onset field along `direction` (Tesla).
pub fn onset_field_along(h_rot: &[f64; 3], direction: &[f64; 3]) -> Result<f64, String> {
    critical_field_along(1.0, h_rot, 0.0, direction)
}

/// d0 . d_{p_i} after field-induced rotation: the orthogonal part shrinks as
/// 1/h_rot above the onset.
pub fn dvector_overlaps(
    field: &[f64; 3],
    h_rot: &[f64; 3],
    overlap0: &[f64; 3],
) -> Result<[f64; 3], String> {
    if !overlap0.iter().all(|c| (0.0..=1.0).contains(c)) {
        return Err(format!(
            "overlap0 entries must lie in [0, 1], got {overlap0:?}"
        ));
    }
    let h = reduced_field(field, h_rot)?;
    let shrink = if h <= 1.0 { 1.0 } else { 1.0 / h };
    Ok(overlap0.map(|c0| {
        let s = (1.0 - c0 * c0).sqrt() * shrink;
        (1.0 - s * s).sqrt()
    }))
}

fn regime_from_flags(uniform: bool, pdw: bool) -> PdwRegime {
    match (uniform, pdw) {
        (true, true) => PdwRegime::Coexistence,
        (false, true) => PdwRegime::PdwOnly,
        (true, false) => PdwRegime::UniformOnly,
        (false, false) => PdwRegime::Normal,
    }
}

/// Name the ordered phase from the two normalised amplitudes.
pub fn classify_regime(delta0: f64, delta_p: f64) -> PdwRegime {
    regime_from_flags(delta0 > 0.0, delta_p > 0.0)
}

fn validate(cfg: &PdwConfig) -> Result<(), String> {
    if !cfg.p_vectors.iter().flatten().all(|v| v.is_finite()) {
        return Err("p_vectors must be finite".to_string());
    }
    if !cfg.temperature.is_finite() || cfg.temperature < 0.0 {
        return Err(format!(
            "temperature must be finite and >= 0, got {}",
            cfg.temperature
        ));
    }
    if !positive(cfg.tc) || !positive(cfg.t_pdw) {
        return Err(format!(
            "tc and t_pdw must be positive, got {} and {}",
            cfg.tc, cfg.t_pdw
        ));
    }
    if !cfg.field.iter().chain(&cfg.phases).all(|v| v.is_finite()) {
        return Err("field and phases must be finite".to_string());
    }
    check_positive_triple("hc2", &cfg.hc2)?;
    check_positive_triple("h_pdw", &cfg.h_pdw)?;
    check_positive_triple("h_rot", &cfg.h_rot)?;
    if !non_negative(cfg.delta0_mev) || !non_negative(cfg.pdw_ratio) {
        return Err("delta0_mev and pdw_ratio must be >= 0".to_string());
    }
    if !cfg.overlap0.iter().all(|c| (0.0..=1.0).contains(c)) {
        return Err(format!(
            "overlap0 entries must lie in [0, 1], got {:?}",
            cfg.overlap0
        ));
    }
    if ![cfg.kappa_p, cfg.kappa_q, cfg.kappa_h, cfg.kappa_harm]
        .iter()
        .all(|k| k.is_finite())
    {
        return Err("kappa couplings must be finite".to_string());
    }
    if !non_negative(cfg.floor) {
        return Err(format!("floor must be >= 0, got {}", cfg.floor));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Composite orders
// ---------------------------------------------------------------------------

/// Evaluate the mean-field amplitudes and every composite order.
pub fn compute_pdw_state(cfg: &PdwConfig) -> Result<PdwState, String> {
    validate(cfg)?;
    let t = cfg.temperature;
    let d0 = landau_amplitude(t, cfg.tc, reduced_field(&cfg.field, &cfg.hc2)?)?;
    let dp = landau_amplitude(t, cfg.t_pdw, reduced_field(&cfg.field, &cfg.h_pdw)?)?;
    let c = dvector_overlaps(&cfg.field, &cfg.h_rot, &cfg.overlap0)?;
    let s = c.map(|ci| (1.0 - ci * ci).max(0.0).sqrt());
    let phase = cfg.phases.map(|phi| Complex64::new(phi.cos(), phi.sin()));

    // Stationary points of F in the CDW amplitudes: rho = -(source)* / r.
    let rho_p: [Complex64; 3] =
        std::array::from_fn(|i| phase[i] * (-2.0 * cfg.kappa_p * d0 * dp * c[i]));
    let rho_q: [Complex64; 3] = std::array::from_fn(|i| {
        phase[i] * phase[i] * (-cfg.kappa_q * (dp * dp)) - rho_p[i] * rho_p[i] * cfg.kappa_harm
    });
    let rho_h = [
        rho_q[2] * -cfg.kappa_h * rho_q[1].conj(),
        rho_q[2] * -cfg.kappa_h * rho_q[0].conj(),
    ];
    let spin_p: [Complex64; 3] =
        std::array::from_fn(|i| phase[i] * (2.0 * cfg.kappa_p * d0 * dp * s[i]));

    Ok(PdwState {
        delta0: d0,
        delta_p: dp,
        overlap: c,
        regime: classify_regime(d0, dp),
        rho_p,
        rho_q,
        rho_h,
        spin_p,
    })
}

/// (label, family, k, rho_k) for every composite CDW component, in the order
/// p1-3, q1-3, h1-2 shared with the Python twin.
fn composite_terms(
    cfg: &PdwConfig,
    state: &PdwState,
) -> [(&'static str, PdwFamily, [f64; 2], Complex64); 8] {
    let p = &cfg.p_vectors;
    let h = harmonic_vectors(p);
    let q = |i: usize| [2.0 * p[i][0], 2.0 * p[i][1]];
    [
        ("p1", PdwFamily::P, p[0], state.rho_p[0]),
        ("p2", PdwFamily::P, p[1], state.rho_p[1]),
        ("p3", PdwFamily::P, p[2], state.rho_p[2]),
        ("q1", PdwFamily::Q, q(0), state.rho_q[0]),
        ("q2", PdwFamily::Q, q(1), state.rho_q[1]),
        ("q3", PdwFamily::Q, q(2), state.rho_q[2]),
        ("h1", PdwFamily::H, h[0], state.rho_h[0]),
        ("h2", PdwFamily::H, h[1], state.rho_h[1]),
    ]
}

/// The eight composite CDW peaks (p1-3, q1-3, h1-2) with amplitudes.
pub fn pdw_peaks(cfg: &PdwConfig, state: &PdwState) -> Vec<PdwPeak> {
    composite_terms(cfg, state)
        .into_iter()
        .map(|(label, family, k, rho)| {
            let amplitude = rho.norm();
            PdwPeak {
                label,
                family,
                kx: k[0],
                ky: k[1],
                amplitude,
                visible: amplitude * amplitude > cfg.floor,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Real-space fields
// ---------------------------------------------------------------------------

/// Evaluate (|d|, rho, S_z) at one point. The pair field is
/// d(r) = Delta_0 n_0 + sum_i 2 Delta_p n_i cos(p_i.r + phi_i) with n_0 = (1, 0)
/// and n_i = (c_i, s_i); densities are sum_k 2 Re[rho_k exp(i k.r)].
pub fn pdw_fields_at(cfg: &PdwConfig, state: &PdwState, x: f64, y: f64) -> (f64, f64, f64) {
    let delta0 = cfg.delta0_mev * state.delta0;
    let delta_p = cfg.delta0_mev * cfg.pdw_ratio * state.delta_p;

    let mut d_par = delta0;
    let mut d_perp = 0.0;
    let mut spin = 0.0;
    for i in 0..3 {
        let p = cfg.p_vectors[i];
        let arg = p[0] * x + p[1] * y;
        let wave = (arg + cfg.phases[i]).cos();
        let s = (1.0 - state.overlap[i] * state.overlap[i]).max(0.0).sqrt();
        d_par += 2.0 * delta_p * state.overlap[i] * wave;
        d_perp += 2.0 * delta_p * s * wave;
        spin += 2.0 * (state.spin_p[i] * Complex64::new(arg.cos(), arg.sin())).re;
    }

    let mut charge = 0.0;
    for (_label, _family, k, rho) in composite_terms(cfg, state) {
        let arg = k[0] * x + k[1] * y;
        charge += 2.0 * (rho * Complex64::new(arg.cos(), arg.sin())).re;
    }

    (d_par.hypot(d_perp), charge, spin)
}

/// Pair amplitude, charge density and spin density on the NxN grid spanning
/// [-extent/2, extent/2] (the moire grid convention).
pub fn pdw_real_space(
    cfg: &PdwConfig,
    state: &PdwState,
    resolution: usize,
    physical_extent: f64,
) -> Result<PdwFields, String> {
    if resolution < 2 {
        return Err(format!("resolution must be >= 2, got {resolution}"));
    }
    if !positive(physical_extent) {
        return Err(format!(
            "physical_extent must be positive, got {physical_extent}"
        ));
    }
    let n = resolution;
    let rows: Vec<Vec<(f64, f64, f64)>> = (0..n)
        .into_par_iter()
        .map(|iy| {
            let y = (iy as f64 / (n - 1) as f64 - 0.5) * physical_extent;
            (0..n)
                .map(|ix| {
                    let x = (ix as f64 / (n - 1) as f64 - 0.5) * physical_extent;
                    pdw_fields_at(cfg, state, x, y)
                })
                .collect()
        })
        .collect();

    let mut pair_amplitude = Vec::with_capacity(n * n);
    let mut charge_density = Vec::with_capacity(n * n);
    let mut spin_density = Vec::with_capacity(n * n);
    for (pair, charge, spin) in rows.into_iter().flatten() {
        pair_amplitude.push(pair);
        charge_density.push(charge);
        spin_density.push(spin);
    }
    Ok(PdwFields {
        pair_amplitude,
        charge_density,
        spin_density,
        resolution: n,
        physical_extent,
    })
}

// ---------------------------------------------------------------------------
// Sweeps and phase diagram
// ---------------------------------------------------------------------------

fn sweep(configs: &[PdwConfig], values: &[f64]) -> Result<PdwSweep, String> {
    let states = configs
        .iter()
        .map(compute_pdw_state)
        .collect::<Result<Vec<_>, _>>()?;
    let column = |f: &dyn Fn(&PdwState) -> f64| states.iter().map(f).collect::<Vec<f64>>();
    Ok(PdwSweep {
        values: values.to_vec(),
        delta0: column(&|s| s.delta0),
        delta_p: column(&|s| s.delta_p),
        intensity_p: std::array::from_fn(|i| column(&|s| s.rho_p[i].norm_sqr())),
        intensity_q: std::array::from_fn(|i| column(&|s| s.rho_q[i].norm_sqr())),
        intensity_h: std::array::from_fn(|j| column(&|s| s.rho_h[j].norm_sqr())),
    })
}

/// Sweep temperature at the configured field.
pub fn temperature_sweep(cfg: &PdwConfig, temperatures: &[f64]) -> Result<PdwSweep, String> {
    let configs: Vec<PdwConfig> = temperatures
        .iter()
        .map(|&temperature| PdwConfig {
            temperature,
            ..*cfg
        })
        .collect();
    sweep(&configs, temperatures)
}

/// Sweep |B| along `direction` at the configured temperature.
pub fn field_sweep(
    cfg: &PdwConfig,
    magnitudes: &[f64],
    direction: &[f64; 3],
) -> Result<PdwSweep, String> {
    let u = unit(direction)?;
    let configs: Vec<PdwConfig> = magnitudes
        .iter()
        .map(|&b| PdwConfig {
            field: u.map(|ui| b * ui),
            ..*cfg
        })
        .collect();
    sweep(&configs, magnitudes)
}

/// Regime on a (T, |B|) grid along `direction`, row-major `t_idx * n_b + b_idx`.
pub fn regime_map(
    cfg: &PdwConfig,
    temperatures: &[f64],
    magnitudes: &[f64],
    direction: &[f64; 3],
) -> Result<Vec<PdwRegime>, String> {
    validate(cfg)?;
    let u = unit(direction)?;
    let k_sc = ellipsoidal_norm(&u, &cfg.hc2);
    let k_pdw = ellipsoidal_norm(&u, &cfg.h_pdw);
    Ok(temperatures
        .iter()
        .flat_map(|&t| {
            magnitudes.iter().map(move |&b| {
                regime_from_flags(
                    1.0 - t / cfg.tc - b.abs() * k_sc > 0.0,
                    1.0 - t / cfg.t_pdw - b.abs() * k_pdw > 0.0,
                )
            })
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    const X_AXIS: [f64; 3] = [1.0, 0.0, 0.0];
    const Y_AXIS: [f64; 3] = [0.0, 1.0, 0.0];
    const Z_AXIS: [f64; 3] = [0.0, 0.0, 1.0];

    fn state_at(temperature: f64, field: [f64; 3]) -> PdwState {
        compute_pdw_state(&PdwConfig {
            temperature,
            field,
            ..Default::default()
        })
        .unwrap()
    }

    fn visible(temperature: f64, field: [f64; 3]) -> Vec<&'static str> {
        let cfg = PdwConfig {
            temperature,
            field,
            ..Default::default()
        };
        let state = compute_pdw_state(&cfg).unwrap();
        pdw_peaks(&cfg, &state)
            .into_iter()
            .filter(|p| p.visible)
            .map(|p| p.label)
            .collect()
    }

    fn norm2(k: [f64; 2]) -> f64 {
        (k[0] * k[0] + k[1] * k[1]).sqrt()
    }

    #[test]
    fn test_ute2_wavevector_relations() {
        let q_te = 2.0 * PI / A_TE_UTE2;
        let p = ute2_wavevectors(A_TE_UTE2).unwrap();
        assert!((norm2(p[2]) / q_te - 0.28).abs() < 1e-12);
        assert!((p[0][0] + p[1][0]).abs() < 1e-15 && (p[0][1] - p[1][1]).abs() < 1e-15);
        let h = harmonic_vectors(&p);
        // h1 = q3 - q2, h2 = q3 - q1, and h1 + h2 + p3 = 0 (hexagonal triple).
        assert!((h[0][0] - (2.0 * p[2][0] - 2.0 * p[1][0])).abs() < 1e-15);
        assert!((h[1][1] - (2.0 * p[2][1] - 2.0 * p[0][1])).abs() < 1e-15);
        assert!((h[0][0] + h[1][0] + p[2][0]).abs() < 1e-12);
        assert!((h[0][1] + h[1][1] + p[2][1]).abs() < 1e-12);
    }

    #[test]
    fn test_moire_locked_wavevectors() {
        let p = moire_locked_wavevectors(36.0, 0.0).unwrap();
        for k in p {
            assert!((norm2(k) - 2.0 * PI / 36.0).abs() < 1e-12);
        }
        let dot = (p[0][0] * p[1][0] + p[0][1] * p[1][1]) / (2.0 * PI / 36.0).powi(2);
        assert!((dot + 0.5).abs() < 1e-12);
        for bad in [0.0, -5.0, f64::INFINITY, f64::NAN] {
            assert!(moire_locked_wavevectors(bad, 0.0).is_err());
        }
    }

    #[test]
    fn test_landau_amplitude_limits() {
        assert_eq!(landau_amplitude(0.0, 5.0, 0.0).unwrap(), 1.0);
        assert_eq!(landau_amplitude(5.0, 5.0, 0.0).unwrap(), 0.0);
        let a = landau_amplitude(1.0, 4.0, 0.25).unwrap();
        assert!((a * a - 0.5).abs() < 1e-12);
        assert!(landau_amplitude(1.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn test_reduced_field_ellipsoidal() {
        assert!((reduced_field(&[3.0, 0.0, 0.0], &[6.0, 1.0, 1.0]).unwrap() - 0.5).abs() < 1e-15);
        assert!((reduced_field(&[3.0, 4.0, 0.0], &[10.0; 3]).unwrap() - 0.5).abs() < 1e-15);
        assert!(reduced_field(&[1.0; 3], &[1.0, 0.0, 1.0]).is_err());
    }

    #[test]
    fn test_regimes() {
        assert_eq!(state_at(1.0, [0.0; 3]).regime, PdwRegime::Coexistence);
        let cheshire = state_at(3.0, [0.0; 3]);
        assert_eq!(cheshire.regime, PdwRegime::PdwOnly);
        assert_eq!(cheshire.delta0, 0.0);
        assert!(cheshire.delta_p > 0.0);
        assert_eq!(state_at(5.5, [0.0; 3]).regime, PdwRegime::Normal);
        // Along a the PDW critical field (3.75 T) sits below Hc2 (11 T).
        assert_eq!(
            state_at(0.3, [0.0, 5.0, 0.0]).regime,
            PdwRegime::UniformOnly
        );
    }

    #[test]
    fn test_p_cdw_needs_uniform_sc_and_q_survives() {
        let above = state_at(3.0, [0.0; 3]);
        assert!(above.rho_p.iter().all(|r| r.norm() == 0.0));
        for r in above.rho_q {
            assert!((r.norm() - above.delta_p * above.delta_p).abs() < 1e-15);
        }
        assert!(state_at(1.0, [0.0; 3]).rho_p[2].norm() > 0.0);
    }

    #[test]
    fn test_q_continuous_across_tc() {
        let below = state_at(2.1 - 1e-9, [0.0; 3]);
        let above = state_at(2.1 + 1e-9, [0.0; 3]);
        for i in 0..3 {
            assert!((below.rho_q[i].norm() - above.rho_q[i].norm()).abs() < 1e-8);
        }
    }

    #[test]
    fn test_harmonic_is_product_of_q() {
        let s = state_at(1.0, [0.0; 3]);
        assert!((s.rho_h[0].norm() - (s.rho_q[2] * s.rho_q[1]).norm()).abs() < 1e-15);
        assert!((s.rho_h[1].norm() - (s.rho_q[2] * s.rho_q[0]).norm()).abs() < 1e-15);
    }

    #[test]
    fn test_sliding_phase_doubles_for_q() {
        let s = compute_pdw_state(&PdwConfig {
            temperature: 1.0,
            phases: [0.0, 0.0, 0.3],
            ..Default::default()
        })
        .unwrap();
        assert!(((-s.rho_p[2]).arg() - 0.3).abs() < 1e-12);
        assert!(((-s.rho_q[2]).arg() - 0.6).abs() < 1e-12);
    }

    #[test]
    fn test_dvector_rotation() {
        assert_eq!(state_at(0.3, [0.0; 3]).overlap, [0.0, 0.0, 1.0]);
        let c = dvector_overlaps(&[0.0, 0.0, 9.0], &H_ROT_UTE2, &OVERLAP0_UTE2).unwrap();
        assert!((c[0] - 0.75_f64.sqrt()).abs() < 1e-12);
        assert_eq!(c[2], 1.0);
        assert!((onset_field_along(&H_ROT_UTE2, &Y_AXIS).unwrap() - 0.4).abs() < 1e-12);
        assert!((onset_field_along(&H_ROT_UTE2, &Z_AXIS).unwrap() - 4.5).abs() < 1e-12);
        for bz in [0.0, 4.0, 6.0, 12.0] {
            let s = state_at(0.3, [0.0, 0.0, bz]);
            let total = (2.0 * s.delta0 * s.delta_p).powi(2);
            for i in 0..3 {
                let sum = s.rho_p[i].norm_sqr() + s.spin_p[i].norm_sqr();
                assert!((sum - total).abs() < 1e-12, "bz = {bz}, i = {i}");
            }
        }
    }

    #[test]
    fn test_zhu_q_suppression_windows_and_hierarchy() {
        let cfg = PdwConfig::default();
        let windows = [(X_AXIS, 1.5, 3.0), (Y_AXIS, 0.4, 0.8), (Z_AXIS, 4.0, 8.0)];
        let mut h_sup = [0.0; 3];
        for (i, (axis, lo, hi)) in windows.iter().enumerate() {
            h_sup[i] = critical_field_along(cfg.t_pdw, &cfg.h_pdw, 4.2, axis).unwrap();
            assert!(*lo < h_sup[i] && h_sup[i] < *hi, "axis {i}: {}", h_sup[i]);
        }
        // b* > c* > a, mirroring the Hc2 hierarchy.
        assert!(h_sup[2] > h_sup[0] && h_sup[0] > h_sup[1]);
    }

    #[test]
    fn test_zhu_temperature_milestones() {
        let zero = [0.0; 3];
        assert_eq!(visible(0.3, zero), ["p3", "q1", "q2", "q3", "h1", "h2"]);
        assert!(visible(2.0, zero).contains(&"p3"));
        assert!(!visible(2.15, zero).contains(&"p3"));
        assert!(visible(3.3, zero).contains(&"h1"));
        assert!(!visible(3.6, zero).contains(&"h1"));
        assert!(visible(4.2, zero).contains(&"q3"));
        assert!(visible(4.8, zero).is_empty());
    }

    #[test]
    fn test_zhu_field_induced_p2() {
        assert!(!visible(0.3, [0.0, 0.0, 4.0]).contains(&"p2"));
        assert!(visible(0.3, [0.0, 0.0, 5.0]).contains(&"p2"));
        assert!(!visible(0.3, [0.0, 0.3, 0.0]).contains(&"p2"));
        assert!(visible(0.3, [0.0, 0.5, 0.0]).contains(&"p2"));
        // 1.6 K window along b*: 5-6 T yes, 8 T no; nothing above Tc.
        assert!(visible(1.6, [0.0, 0.0, 6.0]).contains(&"p2"));
        assert!(!visible(1.6, [0.0, 0.0, 8.0]).contains(&"p2"));
        assert!(!visible(2.5, [0.0, 0.0, 6.0]).contains(&"p2"));
        // Along c*, p2 strengthens up to 2 T and weakens by 3 T.
        let p2 = |bx: f64| state_at(0.3, [bx, 0.0, 0.0]).rho_p[1].norm_sqr();
        assert!(p2(2.0) > p2(0.5) && p2(2.0) > p2(3.0));
    }

    #[test]
    fn test_real_space_grid_and_origin() {
        let cfg = PdwConfig::default();
        let state = compute_pdw_state(&cfg).unwrap();
        let fields = pdw_real_space(&cfg, &state, 33, 80.0).unwrap();
        assert_eq!(fields.pair_amplitude.len(), 33 * 33);
        assert!(fields
            .pair_amplitude
            .iter()
            .all(|&v| v >= 0.0 && v.is_finite()));
        // Odd resolution puts the grid centre on the origin.
        let centre = 16 * 33 + 16;
        let (pair, charge, spin) = pdw_fields_at(&cfg, &state, 0.0, 0.0);
        assert_eq!(fields.pair_amplitude[centre], pair);
        assert_eq!(fields.charge_density[centre], charge);
        assert_eq!(fields.spin_density[centre], spin);
        assert!(pdw_real_space(&cfg, &state, 1, 80.0).is_err());
        assert!(pdw_real_space(&cfg, &state, 8, 0.0).is_err());
    }

    #[test]
    fn test_normal_state_fields_vanish() {
        let cfg = PdwConfig {
            temperature: 6.0,
            ..Default::default()
        };
        let state = compute_pdw_state(&cfg).unwrap();
        let fields = pdw_real_space(&cfg, &state, 16, 40.0).unwrap();
        assert!(fields.pair_amplitude.iter().all(|&v| v == 0.0));
        assert!(fields.charge_density.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_sweeps() {
        let temps: Vec<f64> = (0..=60).map(|i| i as f64 * 0.1).collect();
        let sw = temperature_sweep(&PdwConfig::default(), &temps).unwrap();
        assert_eq!(sw.intensity_q[2].len(), temps.len());
        for (t, ip) in temps.iter().zip(&sw.intensity_p[2]) {
            if *t >= 2.1 {
                assert_eq!(*ip, 0.0);
            }
        }
        assert!(sw.intensity_q[2].windows(2).all(|w| w[1] <= w[0] + 1e-15));

        let mags: Vec<f64> = (0..=10).map(f64::from).collect();
        let a = field_sweep(&PdwConfig::default(), &mags, &[0.0, 0.0, 1.0]).unwrap();
        let b = field_sweep(&PdwConfig::default(), &mags, &[0.0, 0.0, 7.0]).unwrap();
        for (x, y) in a.intensity_p[1].iter().zip(&b.intensity_p[1]) {
            assert!((x - y).abs() < 1e-12);
        }
        assert!(field_sweep(&PdwConfig::default(), &mags, &[0.0; 3]).is_err());
    }

    #[test]
    fn test_regime_map_matches_state() {
        let cfg = PdwConfig::default();
        let temps: Vec<f64> = (0..13).map(|i| i as f64 * 0.5).collect();
        let mags: Vec<f64> = (0..7).map(|i| i as f64 * 2.0).collect();
        let map = regime_map(&cfg, &temps, &mags, &Y_AXIS).unwrap();
        assert_eq!(map.len(), temps.len() * mags.len());
        for (i, &t) in temps.iter().enumerate() {
            for (j, &b) in mags.iter().enumerate() {
                assert_eq!(map[i * mags.len() + j], state_at(t, [0.0, b, 0.0]).regime);
            }
        }
    }

    #[test]
    fn test_moire_locked_period_limits() {
        // theta = 0 is the mismatch formula; equal lattices the twist formula.
        let l0 = moire_locked_period(3.82, 4.264, 0.0).unwrap();
        let mismatch = crate::moire::moire_periodicity_1d(3.82, 4.264);
        assert!((l0 - mismatch).abs() < 1e-12 * mismatch);
        let twist = crate::moire::moire_periodicity_twist(2.46, 1.0);
        let lt = moire_locked_period(2.46, 2.46, 1.0).unwrap();
        assert!((lt - twist).abs() < 1e-9 * twist);
        assert!(moire_locked_period(2.46, 2.46, 0.0).unwrap().is_infinite());
        assert!(moire_locked_period(0.0, 2.46, 1.0).is_err());
        assert!(moire_locked_pdw_ratio(f64::NAN, 20.0, 2.58, 3.6).is_err());
    }

    #[test]
    fn test_moire_locked_preset() {
        let ratio = moire_locked_pdw_ratio(36.0, 20.0, 2.58, 3.60).unwrap();
        let cfg = PdwConfig::moire_locked(36.0, 0.0, (2.58 + 3.60) / 2.0, ratio).unwrap();
        assert_eq!(cfg.t_pdw, cfg.tc);
        // Imprinted CPDM: nothing survives above Tc.
        let hot = compute_pdw_state(&PdwConfig {
            temperature: 1.1 * cfg.tc,
            ..cfg
        })
        .unwrap();
        assert_eq!(hot.regime, PdwRegime::Normal);
        // What-if intrinsic PDW: the q = 2Q_m CDW survives above Tc.
        let what_if = compute_pdw_state(&PdwConfig {
            temperature: 1.2 * TC_FETE_FILM,
            t_pdw: 1.5 * TC_FETE_FILM,
            ..cfg
        })
        .unwrap();
        assert_eq!(what_if.regime, PdwRegime::PdwOnly);
        assert!(what_if.rho_q.iter().all(|r| r.norm() > 0.0));
        // Singlet-like: no field-induced rotation, no composite SDW.
        let fielded = compute_pdw_state(&PdwConfig {
            field: [5.0; 3],
            ..cfg
        })
        .unwrap();
        assert_eq!(fielded.overlap, [1.0; 3]);
        assert!(fielded.spin_p.iter().all(|s| s.norm() == 0.0));
        assert!(PdwConfig::moire_locked(f64::INFINITY, 0.0, 3.09, ratio).is_err());
    }

    #[test]
    fn test_invalid_configs() {
        let base = PdwConfig::default();
        let cases = [
            PdwConfig {
                temperature: -1.0,
                ..base
            },
            PdwConfig {
                temperature: f64::NAN,
                ..base
            },
            PdwConfig { tc: 0.0, ..base },
            PdwConfig {
                t_pdw: -2.0,
                ..base
            },
            PdwConfig {
                hc2: [20.0, 0.0, 34.0],
                ..base
            },
            PdwConfig {
                h_pdw: [14.0, 3.75, -1.0],
                ..base
            },
            PdwConfig {
                h_rot: [0.0, 0.4, 4.5],
                ..base
            },
            PdwConfig {
                field: [0.0, f64::INFINITY, 0.0],
                ..base
            },
            PdwConfig {
                overlap0: [0.0, 1.5, 1.0],
                ..base
            },
            PdwConfig {
                overlap0: [-0.1, 0.0, 1.0],
                ..base
            },
            PdwConfig {
                delta0_mev: -0.1,
                ..base
            },
            PdwConfig {
                pdw_ratio: -0.1,
                ..base
            },
            PdwConfig {
                floor: -1.0,
                ..base
            },
            PdwConfig {
                kappa_q: f64::NAN,
                ..base
            },
            PdwConfig {
                p_vectors: [[f64::NAN, 0.0], [0.0; 2], [0.0; 2]],
                ..base
            },
        ];
        for cfg in &cases {
            assert!(compute_pdw_state(cfg).is_err(), "cfg = {cfg:?}");
        }
    }

    // --- Cross-stack parity anchors (pinned identically in tests/test_pdw.py) ---

    #[test]
    fn test_parity_q_suppression_fields() {
        let cfg = PdwConfig::default();
        let at = |axis: &[f64; 3]| critical_field_along(cfg.t_pdw, &cfg.h_pdw, 4.2, axis).unwrap();
        assert!((at(&X_AXIS) - 2.24).abs() < 1e-12);
        assert!((at(&Y_AXIS) - 0.6).abs() < 1e-12);
        assert!((at(&Z_AXIS) - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_parity_field_induced_p2_amplitude() {
        let s = state_at(0.3, [0.0, 0.0, 6.0]);
        assert!((s.rho_p[1].norm() - 0.963907488848813).abs() < 1e-12);
    }

    #[test]
    fn test_parity_real_space_origin() {
        let cfg = PdwConfig::default();
        let state = compute_pdw_state(&cfg).unwrap();
        let (pair, charge, spin) = pdw_fields_at(&cfg, &state, 0.0, 0.0);
        assert!((pair - 0.607885168114487).abs() < 1e-12, "pair = {pair}");
        assert!(
            (charge - -12.764863559406859).abs() < 1e-12,
            "charge = {charge}"
        );
        assert!((spin - 7.180927118813718).abs() < 1e-12, "spin = {spin}");
    }

    #[test]
    fn test_parity_moire_locked_twisted() {
        // FeTe / Sb2Te3 at a 1 deg twist, T = 4 K, default gaps and xi = 20 A.
        let period = moire_locked_period(3.82, 4.264, 1.0).unwrap();
        assert!(
            (period - 36.23263757853417).abs() < 1e-12,
            "period = {period}"
        );
        let ratio = moire_locked_pdw_ratio(period, 20.0, 2.58, 3.60).unwrap();
        assert!((ratio - 0.047517778919214).abs() < 1e-12, "ratio = {ratio}");
        let cfg = PdwConfig {
            temperature: 4.0,
            ..PdwConfig::moire_locked(period, 1.0, (2.58 + 3.60) / 2.0, ratio).unwrap()
        };
        let state = compute_pdw_state(&cfg).unwrap();
        let (pair, charge, _spin) = pdw_fields_at(&cfg, &state, 0.0, 0.0);
        assert!((pair - 3.331137631734327).abs() < 1e-12, "pair = {pair}");
        assert!(
            (charge - -14.647462277091908).abs() < 1e-12,
            "charge = {charge}"
        );
    }
}
