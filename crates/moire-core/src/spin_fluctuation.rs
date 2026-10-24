//! Spin-fluctuation pairing: LCCO magnetotransport and what-if pairing routes.
//!
//! **SPECULATIVE** module status: LCCO is not one of this project's
//! heterostructures, and Parts B and C are what-ifs.
//!
//! Part A — LCCO transport, paper-faithful (Duffy et al., Nat. Phys. (2026),
//! doi:10.1038/s41567-026-03454-6; arXiv:2502.13612). Boltzmann /
//! Shockley–Chambers magnetotransport of electron-doped La2-xCexCuO4 with the
//! scattering-rate ansatz of Eq. 1, the analytic Fermi surface of Eqs. 7-10
//! and the Jones–Zener Hall coefficient with Kontani current-vertex
//! corrections (Eqs. 11-15). It reproduces the paper's Fig. 4–5 resistivity
//! and magnetoresistance simulations to within about 5–12 %, with one inferred
//! reading, the hot-spot height of Fig. 4b1; its Hall numbers are
//! semi-quantitative (the printed CVC equations do not reproduce Fig. 13).
//! The scattering defines λ_tr, the Fermi-surface-averaged T-linear coupling
//! (Allen's form applied to the microscopic rate); λ_ρ is the Allen–Drude
//! reading of the model's own ρ(T).
//!
//! Part B — SPECULATIVE pairing map: an EMPIRICAL power law through the
//! paper's Fig. 5c Tc labels; the McMillan form is included as a failing
//! comparison (McMillan's formula was derived for phonons).
//!
//! Part C — SPECULATIVE moire glue-modulation what-if. λ(r) = λ̄ (1 + η m~(r))
//! on the project's moire pattern, Tc(r) from Part B and the local gap
//! Δ_loc(r) = Δ̄ Tc(r) / Tc(λ̄), with Δ̄ ∝ Tc*; the map is dressed by the CPDM
//! coherence factor F = exp(-ξ/L_m), with L_m the shared two-lattice period (a
//! convention, so every dressed number is conditional on L_m). Tc(r) = 0
//! regions exist only in the local limit. The LCCO-calibrated map supplies
//! only the *shape* of Tc(λ); it is not a model of FeTe, whose AFM is
//! bicollinear and whose proposed pairing is s± via (π, 0)-type fluctuations.
//!
//! Part D — contrast curve: hole-doped H/T quadrature scaling (Ayres et al.,
//! Nature 595, 661 (2021)), a phenomenological form, not a model.
//!
//! Numerics (SF-spec §5): R2's exact per-cell Chambers integrator (cubic
//! Hermite J/Γ with a spectral derivative, linear Γ/ω, E_n moments), closed
//! by the canonical two-pass recurrence. f′ is computed once per (T, sign B)
//! and the (T, B) pairs run in parallel; every reduction stays sequential, so
//! results do not depend on the thread count.
//!
//! Mirrors `src/waytogocoop/computation/spin_fluctuation.py` in the Python
//! stack — changes here must land in both.

use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner};
use std::f64::consts::PI;
use std::sync::Arc;

use crate::pdw::TC_FETE_FILM;

// ---------------------------------------------------------------------------
// Physical constants and LCCO defaults (Duffy et al. 2026; mirror config.py)
// ---------------------------------------------------------------------------

/// Elementary charge in C (exact SI); Python `ELEMENTARY_CHARGE_C`.
const E_CHARGE: f64 = 1.602176634e-19;
/// Reduced Planck constant in J s; Python `HBAR_J_S`.
const HBAR_J_S: f64 = 1.054571817e-34;
/// ps^-1 -> s^-1.
const PS_INV: f64 = 1.0e12;
/// Ohm m -> micro-Ohm cm.
const OHM_M_TO_MUOHM_CM: f64 = 1.0e8;
/// Angstrom -> m.
const ANG: f64 = 1.0e-10;
/// Largest field or temperature grid the API builds (guards absurd steps).
const MAX_GRID_POINTS: usize = 1_000_000;

/// Boltzmann constant in J/K (exact SI).
pub const KB_J_K: f64 = 1.380649e-23;
/// LCCO in-plane lattice constant in Angstrom (Methods, p. 13).
pub const A_LCCO: f64 = 4.01;
/// LCCO c-axis lattice constant in Angstrom (Methods, p. 13).
pub const C_LCCO: f64 = 12.38;
/// One CuO2 plane per c/2 (inferred: the 0.841-hole Drude R_H, Fig. 4b2 rho
/// and Yuan 2022's A1/d all need it).
pub const LAYER_SPACING_LCCO: f64 = C_LCCO / 2.0;
/// k_F0 in 1/m (Eq. 7, p. 18).
pub const KF0_LCCO: f64 = 5.424e9;
/// alpha in Eq. 7.
pub const KF_ANISOTROPY_LCCO: f64 = 0.11;
/// v_F0 in m/s (Eq. 8).
pub const VF0_LCCO: f64 = 4.946e5;
/// beta in Eq. 8.
pub const VF_ANISOTROPY_LCCO: f64 = 0.03;
/// Impurity rate in ps^-1 (Table 2, x = 0.159, film S1, p. 15).
pub const GAMMA_IMP_LCCO: f64 = 27.0;
/// T-linear rate in ps^-1 K^-1, multiplied by g(x) (Table 2).
pub const ALPHA1_LCCO: f64 = 0.75;
/// T^2 rate in ps^-1 K^-2 (Table 2).
pub const ALPHA2_LCCO: f64 = 0.0112;
/// Effective hot-spot height in ps^-1 plotted in Fig. 4b1 and used in its
/// simulations (inferred; Table 2 prints 50).
pub const HOTSPOT_PEAK_LCCO: f64 = 31.4;
/// Table 2's printed hot-spot height in ps^-1; it does not reproduce Fig. 4.
pub const HOTSPOT_PEAK_TABLE2: f64 = 50.0;
/// Hot-spot 1/e half-width w in rad of exp(-(dphi/w)^2) (Table 2 "HS width").
pub const HOTSPOT_WIDTH_LCCO: f64 = 0.05;
/// Hot-spot height of the Fig. 9c ablation in ps^-1 (p. 14).
pub const HOTSPOT_PEAK_FIG9C: f64 = 105.0;
/// Hot-spot angle in degrees (Fig. 4 caption; the FS meets the AFM zone
/// boundary at 27.20 deg).
pub const PHI_HS_DEG_LCCO: f64 = 27.5;
/// End of the Fermi-surface reconstruction (p. 3).
pub const X_FSR_LCCO: f64 = 0.14;
/// Reference doping, g = 1 (Tables 3-4).
pub const SF_DOPING_REF: f64 = 0.159;
/// g(x): Table 3 (film S1, MR fits at 40 K / 20 T) and Table 4 (S2, from A1).
pub const G_TABLE_LCCO: [(f64, f64); 19] = [
    (0.116, 6.50),
    (0.119, 5.75),
    (0.120, 5.20),
    (0.125, 3.85),
    (0.131, 2.90),
    (0.134, 2.52),
    (0.138, 2.10),
    (0.141, 1.87),
    (0.146, 1.46),
    (0.149, 1.45),
    (0.153, 1.38),
    (0.156, 1.18),
    (0.159, 1.00),
    (0.165, 0.85),
    (0.169, 0.76),
    (0.174, 0.66),
    (0.180, 0.57),
    (0.184, 0.55),
    (0.188, 0.55),
];
/// The 19 tabulated dopings, ascending (Python `DOPINGS_LCCO`).
pub const DOPINGS_LCCO: [f64; 19] = {
    let mut out = [0.0; 19];
    let mut i = 0;
    while i < 19 {
        out[i] = G_TABLE_LCCO[i].0;
        i += 1;
    }
    out
};
/// Fig. 5c top-axis Tc labels (K) on the bottom ticks (vector-located).
pub const TC_LABELS_FIG5C: [(f64, f64); 4] = [(0.12, 25.0), (0.14, 15.0), (0.16, 8.0), (0.18, 0.0)];
/// Tc(x) in K, piecewise linear; the plateau below 0.12 is inferred (Jin
/// 2011, Yuan 2022).
pub const TC_TABLE_LCCO: [(f64, f64); 6] = [
    (0.100, 25.0),
    (0.120, 25.0),
    (0.140, 15.0),
    (0.160, 8.0),
    (0.180, 0.0),
    (0.190, 0.0),
];
/// sin^nu exponent (p. 6, from Grissonnanche et al. 2021).
pub const SF_NU: u32 = 12;
/// Peak of eps(phi), Eq. 13 — Fig. 12a needs ~1; the 1/(1-eps^2) prefactor
/// is dropped (p. 19).
pub const SF_CVC_EPSILON: f64 = 1.0;
/// Width sigma_eps of eps(phi) in rad (Eq. 13, p. 19).
pub const SF_CVC_SIGMA: f64 = 0.07;
/// Fermi-surface samples (a power of two).
pub const SF_N_PHI: usize = 1024;
/// Field grid step in Tesla.
pub const SF_FIELD_STEP: f64 = 0.5;
/// HFML 35 T Bitter magnet (p. 10).
pub const SF_FIELD_MAX: f64 = 35.0;
/// MR family temperatures in K (Fig. 4).
pub const SF_MR_TEMPERATURES: [f64; 4] = [20.0, 30.0, 40.0, 50.0];
/// g(x) was fitted at 40 K and 20 T (Table 3).
pub const SF_FIT_TEMPERATURE: f64 = 40.0;
pub const SF_FIT_FIELD: f64 = 20.0;
/// gamma_1 = d rho / d mu0H at 33 T (Fig. 15).
pub const SF_SLOPE_FIELD: f64 = 33.0;
/// rho0 + A1 T + A2 T^2 fit window in K (Fig. 6 caption).
pub const SF_RHO_FIT_RANGE: (f64, f64) = (15.0, 60.0);
/// Fig. 13 (a)-(d) R_H in 1e-10 m^3/C, T not stated (p. 21); (d)/(a) = 0.494.
pub const SF_FIG13_RH: [f64; 4] = [8.7, 5.6, 10.3, 4.3];

// Part B (SPECULATIVE): pairing maps calibrated to the Fig. 5c labels.
pub const SF_POWER_LAW_LAMBDA_C: f64 = 0.30462015776911294;
pub const SF_POWER_LAW_EXPONENT: f64 = 0.3926512381199364;
/// lambda_tr at x = 0.12.
pub const SF_POWER_LAW_LAMBDA_REF: f64 = 2.1714302476009117;
/// K.
pub const SF_POWER_LAW_TC_REF: f64 = 25.0;
/// McMillan best fit in K; mu* is pinned at 0 (the form fails).
pub const SF_MCMILLAN_OMEGA_K: f64 = 178.04201155099287;
pub const SF_MCMILLAN_MU_STAR: f64 = 0.0;

// Part C (SPECULATIVE): moire glue what-if.
/// Depth of the what-if glue modulation (illustrative).
pub const SF_GLUE_ETA: f64 = 0.10;
/// xi at TC_FETE_FILM in Angstrom; Python `config.DEFAULT_COHERENCE_LENGTH`.
pub const SF_COHERENCE_LENGTH_FILM: f64 = 20.0;
/// Default L_m in Angstrom: the shared two-lattice period of FeTe / Sb2Te3 at
/// 0 deg, pinned so `Default` stays infallible. Equal (bitwise) to
/// `crate::pdw::moire_locked_period(3.82, 4.264, 0.0)`; a test asserts it.
/// It is a convention: the default pattern's visible beat is about 111 A, and
/// every dressed glue number is conditional on L_m.
pub const SF_GLUE_MOIRE_PERIOD: f64 = 36.68576576576573;
/// Measured FeTe-film gaps in meV; Python `config.DELTA_1` / `DELTA_2`.
const DELTA_1: f64 = 2.58;
const DELTA_2: f64 = 3.60;
const DELTA_AVG: f64 = (DELTA_1 + DELTA_2) / 2.0;
const DELTA_AMPLITUDE: f64 = (DELTA_2 - DELTA_1) / 2.0;

// Part D: hole-doped quadrature contrast, Tl2201 (Ayres et al. 2021, ED Fig. 2).
/// micro-Ohm cm.
pub const SF_QUAD_RHO0: f64 = 15.5;
/// micro-Ohm cm / K.
pub const SF_QUAD_A: f64 = 0.14;
/// micro-Ohm cm / K^2.
pub const SF_QUAD_B: f64 = 0.003;
/// micro-Ohm cm / K.
pub const SF_QUAD_ALPHA_KB: f64 = 0.04;
/// micro-Ohm cm / T.
pub const SF_QUAD_GAMMA_MUB: f64 = 0.20;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// Anisotropy W(phi) of the inelastic (g alpha1 T + alpha2 T^2) rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SfInelasticForm {
    /// Paper Eq. 1: two sin^nu branches peaked at the AFM zone boundary.
    #[default]
    AfmBoundary,
    /// W = 1 (Fig. 9c1; the Kohler check).
    Isotropic,
    /// W = cos^2(2 phi), peaked at the antinodes (Fig. 9e1; Gamma_AN of
    /// Fig. 13c).
    Antinodal,
    /// No inelastic term (Fig. 13a/b).
    None,
}

impl SfInelasticForm {
    /// Ordered like Python's `SF_INELASTIC_FORMS`.
    pub const ALL: [SfInelasticForm; 4] = [
        SfInelasticForm::AfmBoundary,
        SfInelasticForm::Isotropic,
        SfInelasticForm::Antinodal,
        SfInelasticForm::None,
    ];

    /// Python `INELASTIC_LABELS`.
    pub fn label(&self) -> &'static str {
        match self {
            SfInelasticForm::AfmBoundary => "AFM zone boundary — paper (Eq. 1)",
            SfInelasticForm::Isotropic => "Isotropic (Fig. 9c)",
            SfInelasticForm::Antinodal => "Antinodal (Fig. 9e, 13c)",
            SfInelasticForm::None => "None (impurity + hot spots only)",
        }
    }

    /// Python identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            SfInelasticForm::AfmBoundary => "afm_boundary",
            SfInelasticForm::Isotropic => "isotropic",
            SfInelasticForm::Antinodal => "antinodal",
            SfInelasticForm::None => "none",
        }
    }
}

/// Which Tc(lambda) map Part B uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SfPairingForm {
    /// Empirical power law through the three superconducting Fig. 5c labels.
    #[default]
    PowerLaw,
    /// McMillan's (phonon) form at its best fit, shown failing.
    McMillan,
}

impl SfPairingForm {
    /// Ordered like Python's `SF_PAIRING_FORMS`.
    pub const ALL: [SfPairingForm; 2] = [SfPairingForm::PowerLaw, SfPairingForm::McMillan];

    /// Python `PAIRING_LABELS`.
    pub fn label(&self) -> &'static str {
        match self {
            SfPairingForm::PowerLaw => "Empirical power law (default)",
            SfPairingForm::McMillan => "McMillan form, best fit — fails",
        }
    }

    /// Python identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            SfPairingForm::PowerLaw => "power_law",
            SfPairingForm::McMillan => "mcmillan",
        }
    }
}

/// Coherence model of the glue what-if: xi(Tc*) = xi_film (TC_FETE_FILM /
/// Tc*)^s and F = exp(-xi / L_m).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum SfCoherence {
    /// s = 1.
    #[default]
    Clean,
    /// s = 1/2.
    Dirty,
    /// No dressing (F = 1); xi is reported with the clean scaling.
    Local,
}

impl SfCoherence {
    /// Ordered like Python's `SF_COHERENCE_MODELS`.
    pub const ALL: [SfCoherence; 3] = [SfCoherence::Clean, SfCoherence::Dirty, SfCoherence::Local];

    /// Python `COHERENCE_LABELS`.
    pub fn label(&self) -> &'static str {
        match self {
            SfCoherence::Clean => "Clean: ξ ∝ 1/Tc (default)",
            SfCoherence::Dirty => "Dirty: ξ ∝ Tc^−½",
            SfCoherence::Local => "Local limit (no coherence)",
        }
    }

    /// Python identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            SfCoherence::Clean => "clean",
            SfCoherence::Dirty => "dirty",
            SfCoherence::Local => "local",
        }
    }
}

/// Doping regime of the LCCO series (SF-spec §3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfDopingRegime {
    /// x < X_FSR_LCCO.
    FsReconstruction,
    /// Tc(x) > 0.
    Superconducting,
    BeyondDome,
}

impl SfDopingRegime {
    /// Ordered like Python's `SF_DOPING_REGIMES`.
    pub const ALL: [SfDopingRegime; 3] = [
        SfDopingRegime::FsReconstruction,
        SfDopingRegime::Superconducting,
        SfDopingRegime::BeyondDome,
    ];

    /// Short display name, as in "Doping regime: Superconducting".
    pub fn label(&self) -> &'static str {
        match self {
            SfDopingRegime::FsReconstruction => "Fermi-surface reconstruction",
            SfDopingRegime::Superconducting => "Superconducting",
            SfDopingRegime::BeyondDome => "Beyond the dome",
        }
    }

    /// Python identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            SfDopingRegime::FsReconstruction => "fs_reconstruction",
            SfDopingRegime::Superconducting => "superconducting",
            SfDopingRegime::BeyondDome => "beyond_dome",
        }
    }

    /// One-line blurb (the web page's `_REGIME_TEXT`).
    pub fn blurb(&self) -> &'static str {
        match self {
            SfDopingRegime::FsReconstruction => {
                "Short-range AFM reconstructs the Fermi surface below x = 0.14; the measured MR \
                 gains a negative low-T component this model omits (g(x) was fitted at 40 K, \
                 where it is weak)."
            }
            SfDopingRegime::Superconducting => {
                "Overdoped and superconducting: only the anisotropic T-linear strength g(x) \
                 varies across the series (Tables 3–4)."
            }
            SfDopingRegime::BeyondDome => {
                "Beyond the dome: Tc = 0 and g(x) has fallen about tenfold from x = 0.116."
            }
        }
    }
}

/// LCCO transport model; defaults are Table 2 at x = 0.159 with the effective
/// hot spot.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SfConfig {
    /// Ce doping x in [0.116, 0.188]; g(x) is interpolated between the table
    /// points.
    pub doping: f64,
    pub inelastic: SfInelasticForm,
    /// Effective AFM hot spots Gamma_HS.
    pub hotspots: bool,
    /// Impurity rate in ps^-1, > 0.
    pub gamma_imp: f64,
    /// T-linear rate in ps^-1 K^-1 (times g(x)), >= 0.
    pub alpha1: f64,
    /// T^2 rate in ps^-1 K^-2, >= 0.
    pub alpha2: f64,
    /// Hot-spot height in ps^-1, >= 0.
    pub hotspot_peak: f64,
    /// Hot-spot 1/e half-width in rad, > 0.
    pub hotspot_width: f64,
    /// Even, 2..=24.
    pub nu: u32,
    /// Hot-spot angle in degrees, in (0, 45).
    pub phi_hs_deg: f64,
    /// 1/m.
    pub kf0: f64,
    /// [0, 1).
    pub kf_anisotropy: f64,
    /// m/s.
    pub vf0: f64,
    /// [0, 1).
    pub vf_anisotropy: f64,
    /// Angstrom, > 0.
    pub layer_spacing: f64,
    /// CVC peak eps0 in [0, 1].
    pub cvc_epsilon: f64,
    /// CVC width in rad, > 0.
    pub cvc_sigma: f64,
    /// Fermi-surface samples: a power of two in [64, 8192].
    pub n_phi: usize,
}

impl Default for SfConfig {
    fn default() -> Self {
        Self {
            doping: SF_DOPING_REF,
            inelastic: SfInelasticForm::AfmBoundary,
            hotspots: true,
            gamma_imp: GAMMA_IMP_LCCO,
            alpha1: ALPHA1_LCCO,
            alpha2: ALPHA2_LCCO,
            hotspot_peak: HOTSPOT_PEAK_LCCO,
            hotspot_width: HOTSPOT_WIDTH_LCCO,
            nu: SF_NU,
            phi_hs_deg: PHI_HS_DEG_LCCO,
            kf0: KF0_LCCO,
            kf_anisotropy: KF_ANISOTROPY_LCCO,
            vf0: VF0_LCCO,
            vf_anisotropy: VF_ANISOTROPY_LCCO,
            layer_spacing: LAYER_SPACING_LCCO,
            cvc_epsilon: SF_CVC_EPSILON,
            cvc_sigma: SF_CVC_SIGMA,
            n_phi: SF_N_PHI,
        }
    }
}

/// **SPECULATIVE** Tc(lambda) map (Part B).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SfPairingMap {
    pub form: SfPairingForm,
    pub lambda_c: f64,
    pub exponent: f64,
    pub lambda_ref: f64,
    /// K.
    pub tc_ref: f64,
    /// K.
    pub omega_k: f64,
    pub mu_star: f64,
}

impl Default for SfPairingMap {
    fn default() -> Self {
        Self {
            form: SfPairingForm::PowerLaw,
            lambda_c: SF_POWER_LAW_LAMBDA_C,
            exponent: SF_POWER_LAW_EXPONENT,
            lambda_ref: SF_POWER_LAW_LAMBDA_REF,
            tc_ref: SF_POWER_LAW_TC_REF,
            omega_k: SF_MCMILLAN_OMEGA_K,
            mu_star: SF_MCMILLAN_MU_STAR,
        }
    }
}

/// **SPECULATIVE** moire glue what-if (Part C).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct SfGlueConfig {
    /// Glue modulation depth, [0, 1).
    pub eta: f64,
    /// Operating Tc* in K, (0, tc_ceiling) and resolvable: Tc(lambda_bar) > 0.
    pub tc_target: f64,
    /// The film's mean gap AT TC_FETE_FILM in meV; Delta_bar scales with
    /// tc_target.
    pub delta_avg: f64,
    pub coherence: SfCoherence,
    /// L_m in Angstrom, finite and > 0.
    pub moire_period: f64,
    pub pairing: SfPairingMap,
}

impl Default for SfGlueConfig {
    fn default() -> Self {
        Self {
            eta: SF_GLUE_ETA,
            tc_target: TC_FETE_FILM,
            delta_avg: DELTA_AVG,
            coherence: SfCoherence::Clean,
            moire_period: SF_GLUE_MOIRE_PERIOD,
            pairing: SfPairingMap::default(),
        }
    }
}

/// Hole-doped quadrature form (Ayres et al. 2021), Tl2201 parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfQuadratureConfig {
    /// micro-Ohm cm.
    pub rho0: f64,
    /// micro-Ohm cm / K.
    pub a_g: f64,
    /// micro-Ohm cm / K^2.
    pub b: f64,
    /// micro-Ohm cm / K.
    pub alpha_kb: f64,
    /// micro-Ohm cm / T.
    pub gamma_mub: f64,
}

impl Default for SfQuadratureConfig {
    fn default() -> Self {
        Self {
            rho0: SF_QUAD_RHO0,
            a_g: SF_QUAD_A,
            b: SF_QUAD_B,
            alpha_kb: SF_QUAD_ALPHA_KB,
            gamma_mub: SF_QUAD_GAMMA_MUB,
        }
    }
}

/// The analytic Fermi surface on the phi_m = 2 pi m / N grid (SF-spec §3.2).
#[derive(Debug, Clone, PartialEq)]
pub struct SfFermiSurface {
    /// rad, polar angle of k - M from the antinodal axis.
    pub phi: Vec<f64>,
    /// 1/m (Eq. 7).
    pub kf: Vec<f64>,
    /// m/s (Eq. 8).
    pub vf: Vec<f64>,
    /// rad (Eq. 10).
    pub gamma: Vec<f64>,
    /// m/s, outward normal (Eq. 9).
    pub vx: Vec<f64>,
    pub vy: Vec<f64>,
    /// DOS weight k_F / (v_F cos gamma) in s/m^2.
    pub weight: Vec<f64>,
    /// e v_F cos gamma / (hbar k_F) in rad s^-1 T^-1.
    pub omega_per_tesla: Vec<f64>,
}

/// MR family; the outer index of every 2D field is the temperature.
#[derive(Debug, Clone, PartialEq)]
pub struct SfMagnetoresistance {
    /// K.
    pub temperatures: Vec<f64>,
    /// T, 0 ..= field_max.
    pub fields: Vec<f64>,
    /// micro-Ohm cm.
    pub rho_xx: Vec<Vec<f64>>,
    /// micro-Ohm cm.
    pub rho_yx: Vec<Vec<f64>>,
    /// rho_xx(B) / rho_xx(0) - 1.
    pub mr: Vec<Vec<f64>>,
    /// micro-Ohm cm / T, second-order stencil (SF-spec §3.4).
    pub drho_dh: Vec<Vec<f64>>,
    /// T / K.
    pub h_over_t: Vec<Vec<f64>>,
}

/// rho0 + A1 T + A2 T^2 least-squares fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfResistivityFit {
    /// micro-Ohm cm.
    pub rho0: f64,
    /// micro-Ohm cm / K.
    pub a1: f64,
    /// micro-Ohm cm / K^2.
    pub a2: f64,
    pub t_min: f64,
    pub t_max: f64,
}

/// One row of the doping sweep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfDopingPoint {
    pub doping: f64,
    pub g: f64,
    /// K, from TC_TABLE_LCCO.
    pub tc: f64,
    pub lambda_tr: f64,
    /// Allen-Drude lambda_rho from the model's own 15-60 K rho(T).
    pub lambda_rho: f64,
    /// rho(0, T) in micro-Ohm cm.
    pub rho0: f64,
    /// MR(field, T).
    pub mr: f64,
}

/// **SPECULATIVE** glue-modulated gap on a moire pattern (Part C).
#[derive(Debug, Clone, PartialEq)]
pub struct SfGlueResult {
    pub lambda_bar: f64,
    /// S = d ln Tc / d ln lambda at lambda_bar.
    pub sensitivity: f64,
    /// xi(Tc*) in Angstrom (the clean value in the local limit).
    pub coherence_length: f64,
    /// F (1.0 in the local limit).
    pub coherence_factor: f64,
    /// L_m in Angstrom.
    pub moire_period: f64,
    pub lambda_map: Vec<f64>,
    /// Local Tc(lambda(r)) in K.
    pub tc_map: Vec<f64>,
    /// Dressed Delta(r) in meV.
    pub gap: Vec<f64>,
    /// Dressed (max - min) / (2 Delta_bar).
    pub relative_modulation: f64,
    pub local_relative_modulation: f64,
    /// mean[tc_map == 0].
    pub local_normal_fraction: f64,
    /// local_normal_fraction in the local limit, else 0.
    pub normal_fraction: f64,
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// `v > 0`, false for NaN (validation must reject NaN, unlike `v <= 0.0`).
fn positive(v: f64) -> bool {
    v > 0.0
}

/// `v >= 0`, false for NaN.
fn non_negative(v: f64) -> bool {
    v >= 0.0
}

fn finite_positive(v: f64) -> bool {
    v.is_finite() && positive(v)
}

fn finite_non_negative(v: f64) -> bool {
    v.is_finite() && non_negative(v)
}

fn validate(cfg: &SfConfig) -> Result<(), String> {
    let (lo, hi) = (G_TABLE_LCCO[0].0, G_TABLE_LCCO[18].0);
    if !(lo - 1e-12..=hi + 1e-12).contains(&cfg.doping) {
        return Err(format!(
            "doping must lie in [{lo}, {hi}], got {}",
            cfg.doping
        ));
    }
    if !finite_positive(cfg.gamma_imp) {
        return Err(format!(
            "gamma_imp must be finite and > 0, got {}",
            cfg.gamma_imp
        ));
    }
    for (name, v) in [
        ("alpha1", cfg.alpha1),
        ("alpha2", cfg.alpha2),
        ("hotspot_peak", cfg.hotspot_peak),
    ] {
        if !finite_non_negative(v) {
            return Err(format!("{name} must be finite and >= 0, got {v}"));
        }
    }
    for (name, v) in [
        ("hotspot_width", cfg.hotspot_width),
        ("cvc_sigma", cfg.cvc_sigma),
        ("kf0", cfg.kf0),
        ("vf0", cfg.vf0),
        ("layer_spacing", cfg.layer_spacing),
    ] {
        if !finite_positive(v) {
            return Err(format!("{name} must be finite and > 0, got {v}"));
        }
    }
    if !(2..=24).contains(&cfg.nu) || !cfg.nu.is_multiple_of(2) {
        return Err(format!(
            "nu must be an even integer in [2, 24], got {}",
            cfg.nu
        ));
    }
    let phi_ok = cfg.phi_hs_deg > 0.0 && cfg.phi_hs_deg < 45.0;
    if !phi_ok {
        return Err(format!(
            "phi_hs_deg must lie in (0, 45), got {}",
            cfg.phi_hs_deg
        ));
    }
    for (name, v) in [
        ("kf_anisotropy", cfg.kf_anisotropy),
        ("vf_anisotropy", cfg.vf_anisotropy),
    ] {
        if !(0.0..1.0).contains(&v) {
            return Err(format!("{name} must lie in [0, 1), got {v}"));
        }
    }
    if !(0.0..=1.0).contains(&cfg.cvc_epsilon) {
        return Err(format!(
            "cvc_epsilon must lie in [0, 1], got {}",
            cfg.cvc_epsilon
        ));
    }
    if !cfg.n_phi.is_power_of_two() || !(64..=8192).contains(&cfg.n_phi) {
        return Err(format!(
            "n_phi must be a power of two in [64, 8192], got {}",
            cfg.n_phi
        ));
    }
    Ok(())
}

fn check_temperature(t: f64) -> Result<(), String> {
    if finite_non_negative(t) {
        Ok(())
    } else {
        Err(format!("temperature must be finite and >= 0, got {t}"))
    }
}

fn check_temperatures(temperatures: &[f64]) -> Result<(), String> {
    temperatures.iter().try_for_each(|&t| check_temperature(t))
}

fn check_fields(fields: &[f64]) -> Result<(), String> {
    match fields.iter().find(|b| !b.is_finite()) {
        Some(b) => Err(format!("fields must be finite, got {b}")),
        None => Ok(()),
    }
}

fn validate_pairing(p: &SfPairingMap) -> Result<(), String> {
    if !finite_non_negative(p.lambda_c) || !p.lambda_ref.is_finite() || p.lambda_ref <= p.lambda_c {
        return Err(format!(
            "power law needs lambda_ref > lambda_c >= 0, got {} and {}",
            p.lambda_ref, p.lambda_c
        ));
    }
    if !finite_positive(p.exponent) || !finite_positive(p.tc_ref) {
        return Err(format!(
            "power law needs exponent > 0 and tc_ref > 0, got {} and {}",
            p.exponent, p.tc_ref
        ));
    }
    if !finite_positive(p.omega_k) || !(0.0..1.0 / 0.62).contains(&p.mu_star) {
        return Err(format!(
            "McMillan needs omega > 0 and 0 <= mu* < 1/0.62, got {} and {}",
            p.omega_k, p.mu_star
        ));
    }
    Ok(())
}

fn check_eta(eta: f64) -> Result<(), String> {
    if (0.0..1.0).contains(&eta) {
        Ok(())
    } else {
        Err(format!("eta must lie in [0, 1), got {eta}"))
    }
}

/// 0 < tc < tc_ceiling(p), and the inverse must not round onto the threshold
/// (Tc(lambda) = 0, which the glue quantities divide by): the default power law
/// resolves Tc* > 6.173098052100439e-06 K. The caller has validated `p`.
/// Mirrors Python's `lambda_from_tc`, which every glue entry point runs.
fn check_tc_target(tc: f64, p: &SfPairingMap) -> Result<(), String> {
    let ceiling = tc_ceiling(p);
    let inside = tc > 0.0 && tc < ceiling;
    if !inside {
        return Err(format!("Tc target must lie in (0, {ceiling}) K, got {tc}"));
    }
    // Negated through `positive` so a NaN lambda is rejected too (McMillan at a
    // denormal tc: 1.45 tc / omega underflows to 0 and ln(0) = -inf, where
    // Python's math.log raises).
    if !positive(tc_from_lambda(lambda_from_tc_unchecked(tc, p), p)) {
        return Err("tc lies below the pairing map's resolvable range: Tc(lambda) = 0".to_string());
    }
    Ok(())
}

fn validate_glue(glue: &SfGlueConfig) -> Result<(), String> {
    validate_pairing(&glue.pairing)?;
    check_eta(glue.eta)?;
    check_tc_target(glue.tc_target, &glue.pairing)?;
    if !finite_positive(glue.delta_avg) {
        return Err(format!(
            "delta_avg must be finite and > 0, got {}",
            glue.delta_avg
        ));
    }
    if !finite_positive(glue.moire_period) {
        return Err(format!(
            "moire_period must be finite and > 0, got {}",
            glue.moire_period
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Part A — LCCO transport (paper-faithful): tables
// ---------------------------------------------------------------------------

/// g(x): linear interpolation of G_TABLE_LCCO, exact at the table points
/// (|dx| <= 1e-12); Err outside [0.116, 0.188].
pub fn g_of_x(doping: f64) -> Result<f64, String> {
    let (lo, hi) = (G_TABLE_LCCO[0].0, G_TABLE_LCCO[18].0);
    if !(lo - 1e-12..=hi + 1e-12).contains(&doping) {
        return Err(format!("doping must lie in [{lo}, {hi}], got {doping}"));
    }
    for pair in G_TABLE_LCCO.windows(2) {
        let ((x0, g0), (x1, g1)) = (pair[0], pair[1]);
        if doping <= x1 + 1e-12 {
            if (doping - x0).abs() <= 1e-12 {
                return Ok(g0);
            }
            if (doping - x1).abs() <= 1e-12 {
                return Ok(g1);
            }
            return Ok(g0 + (g1 - g0) * (doping - x0) / (x1 - x0));
        }
    }
    Ok(G_TABLE_LCCO[18].1)
}

/// Tc(x) in K: piecewise-linear TC_TABLE_LCCO, clamped at >= 0.
pub fn tc_of_x(doping: f64) -> f64 {
    let table = &TC_TABLE_LCCO;
    if doping <= table[0].0 {
        return table[0].1;
    }
    for pair in table.windows(2) {
        let ((x0, t0), (x1, t1)) = (pair[0], pair[1]);
        if doping <= x1 {
            let tc = t0 + (t1 - t0) * (doping - x0) / (x1 - x0);
            return if tc > 0.0 { tc } else { 0.0 };
        }
    }
    table[table.len() - 1].1
}

/// The nearest tabulated doping (ties go to the lower x); non-finite input
/// gives SF_DOPING_REF.
pub fn snap_doping(doping: f64) -> f64 {
    if !doping.is_finite() {
        return SF_DOPING_REF;
    }
    let mut best = DOPINGS_LCCO[0];
    for &x in &DOPINGS_LCCO[1..] {
        if (x - doping).abs() < (best - doping).abs() {
            best = x;
        }
    }
    best
}

/// Doping regime: x < X_FSR -> FS reconstruction; Tc(x) > 0 ->
/// superconducting; otherwise beyond the dome.
pub fn doping_regime(doping: f64) -> SfDopingRegime {
    if doping < X_FSR_LCCO {
        SfDopingRegime::FsReconstruction
    } else if tc_of_x(doping) > 0.0 {
        SfDopingRegime::Superconducting
    } else {
        SfDopingRegime::BeyondDome
    }
}

/// Combinatorial film: "S1" for x <= 0.159 (Table 3), else "S2" (Table 4).
pub fn film_of_x(doping: f64) -> &'static str {
    if doping <= SF_DOPING_REF {
        "S1"
    } else {
        "S2"
    }
}

// ---------------------------------------------------------------------------
// Part A: Fermi surface, scattering rate and CVC geometry
// ---------------------------------------------------------------------------

/// (k_F, v_F, gamma, v_x, v_y) at an arbitrary phi (Eqs. 7-10); the grid and
/// the CVC partner share this routine.
pub fn fermi_surface_at(phi: f64, cfg: &SfConfig) -> (f64, f64, f64, f64, f64) {
    let s = (2.0 * phi).sin();
    let s2 = s * s;
    let kf = cfg.kf0 * (1.0 + cfg.kf_anisotropy * s2);
    let vf = cfg.vf0 * (1.0 + cfg.vf_anisotropy * s2);
    let gamma =
        (2.0 * cfg.kf_anisotropy * (4.0 * phi).sin() / (1.0 + cfg.kf_anisotropy * s2)).atan();
    (
        kf,
        vf,
        gamma,
        vf * (phi - gamma).cos(),
        vf * (phi - gamma).sin(),
    )
}

fn fermi_surface_grid(cfg: &SfConfig) -> SfFermiSurface {
    let n = cfg.n_phi;
    let mut fs = SfFermiSurface {
        phi: Vec::with_capacity(n),
        kf: Vec::with_capacity(n),
        vf: Vec::with_capacity(n),
        gamma: Vec::with_capacity(n),
        vx: Vec::with_capacity(n),
        vy: Vec::with_capacity(n),
        weight: Vec::with_capacity(n),
        omega_per_tesla: Vec::with_capacity(n),
    };
    for m in 0..n {
        let phi = 2.0 * PI * m as f64 / n as f64;
        let (kf, vf, gamma, vx, vy) = fermi_surface_at(phi, cfg);
        let cos_gamma = gamma.cos();
        fs.phi.push(phi);
        fs.kf.push(kf);
        fs.vf.push(vf);
        fs.gamma.push(gamma);
        fs.vx.push(vx);
        fs.vy.push(vy);
        fs.weight.push(kf / (vf * cos_gamma));
        fs.omega_per_tesla
            .push(E_CHARGE * vf * cos_gamma / (HBAR_J_S * kf));
    }
    fs
}

/// The Fermi surface on the n_phi grid.
pub fn fermi_surface(cfg: &SfConfig) -> Result<SfFermiSurface, String> {
    validate(cfg)?;
    Ok(fermi_surface_grid(cfg))
}

/// Luttinger hole count per Cu: a^2 mean(k_F^2) / (2 pi) (0.8392 against
/// 1 - x = 0.841).
pub fn luttinger_holes(cfg: &SfConfig) -> Result<f64, String> {
    validate(cfg)?;
    let fs = fermi_surface_grid(cfg);
    let mean = fs.kf.iter().map(|k| k * k).sum::<f64>() / fs.kf.len() as f64;
    let a = A_LCCO * ANG;
    Ok(a * a * mean / (2.0 * PI))
}

/// Root of k_F(phi)(cos phi + sin phi) = pi/a in (0, pi/4), in radians: a
/// 100-step bisection returning the final midpoint (never via degrees).
fn zone_boundary_crossing(cfg: &SfConfig) -> Result<f64, String> {
    let q = PI / (A_LCCO * ANG);
    let f = |p: f64| fermi_surface_at(p, cfg).0 * (p.cos() + p.sin()) - q;
    let (mut lo, mut hi) = (0.0, PI / 4.0);
    let bracketed = f(lo) < 0.0 && f(hi) > 0.0;
    if !bracketed {
        return Err(
            "the Fermi surface does not cross the AFM zone boundary in (0, pi/4)".to_string(),
        );
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(0.5 * (lo + hi))
}

/// Where the Fermi surface crosses the AFM zone boundary, in degrees
/// (27.197 deg by default); Err when it does not.
pub fn zone_boundary_crossing_deg(cfg: &SfConfig) -> Result<f64, String> {
    validate(cfg)?;
    Ok(zone_boundary_crossing(cfg)?.to_degrees())
}

/// [p, pi/2 - p, p + pi/2, pi - p, p + pi, 3pi/2 - p, p + 3pi/2, 2pi - p].
fn eight_angles(p: f64) -> [f64; 8] {
    let mut out = [0.0; 8];
    for m in 0..4 {
        let shift = m as f64 * 0.5 * PI;
        out[2 * m] = p + shift;
        out[2 * m + 1] = 0.5 * PI - p + shift;
    }
    out
}

/// Gamma_HS centres phi_n in rad (SF-spec §3.3 order).
pub fn hotspot_angles(cfg: &SfConfig) -> [f64; 8] {
    eight_angles(cfg.phi_hs_deg.to_radians())
}

/// eps(phi) centres in rad (D5): the eight-angle pattern on the FS x
/// AFM-zone-boundary crossing. Err when the Fermi surface misses the zone
/// boundary.
pub fn cvc_centres(cfg: &SfConfig) -> Result<[f64; 8], String> {
    validate(cfg)?;
    Ok(eight_angles(zone_boundary_crossing(cfg)?))
}

/// ((d + pi) mod 2 pi) - pi with a positive modulus (np.mod semantics).
fn wrap(d: f64) -> f64 {
    (d + PI).rem_euclid(2.0 * PI) - PI
}

/// sum_n exp(-(wrap(phi - c_n) / width)^2), in centre order.
fn gaussian_sum(phi: f64, centres: &[f64; 8], width: f64) -> f64 {
    let mut out = 0.0;
    for &c in centres {
        let d = wrap(phi - c) / width;
        out += (-(d * d)).exp();
    }
    out
}

/// W(phi) of the inelastic term (SF-spec §3.3, amended D3).
pub fn inelastic_weight(phi: f64, cfg: &SfConfig) -> f64 {
    match cfg.inelastic {
        SfInelasticForm::AfmBoundary => {
            let off = 0.25 * PI - cfg.phi_hs_deg.to_radians();
            let nu = cfg.nu as i32;
            (2.0 * (phi + off)).sin().powi(nu) + (2.0 * (phi - off)).sin().powi(nu)
        }
        SfInelasticForm::Antinodal => (2.0 * phi).cos().powi(2),
        SfInelasticForm::Isotropic => 1.0,
        SfInelasticForm::None => 0.0,
    }
}

/// Gamma in ps^-1 from precomputed W(phi) and hot-spot sum; the one formula
/// behind `scattering_rate` and every transport path.
fn gamma_from_parts(cfg: &SfConfig, g: f64, t: f64, w: f64, hotspot: Option<f64>) -> f64 {
    let amp = g * cfg.alpha1 * t + cfg.alpha2 * (t * t);
    let gam = cfg.gamma_imp + amp * w;
    match hotspot {
        Some(s) => gam + cfg.hotspot_peak * s,
        None => gam,
    }
}

/// Whether Gamma_HS enters (hot spots on with a nonzero height).
fn hotspots_on(cfg: &SfConfig) -> bool {
    cfg.hotspots && cfg.hotspot_peak != 0.0
}

/// Gamma(phi; T, x) in ps^-1: Gamma_imp + Gamma_HS(phi) + (g alpha1 T +
/// alpha2 T^2) W(phi) (Eq. 1 resolved: sum of branches, effective hot spots).
pub fn scattering_rate(phi: f64, temperature: f64, cfg: &SfConfig) -> Result<f64, String> {
    validate(cfg)?;
    check_temperature(temperature)?;
    let g = g_of_x(cfg.doping)?;
    let hotspot =
        hotspots_on(cfg).then(|| gaussian_sum(phi, &hotspot_angles(cfg), cfg.hotspot_width));
    Ok(gamma_from_parts(
        cfg,
        g,
        temperature,
        inelastic_weight(phi, cfg),
        hotspot,
    ))
}

/// sgn with sgn(0) = 0 (np.sign semantics, D8). `f64::signum` would give +1
/// at 0 and move the antinode's partner off pi.
fn sgn(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Geometric CVC partner phi' of phi: k + Q projected onto the pocket (the
/// polar angle about the nearest pocket centre), in [0, 2 pi). phi = 0 maps
/// to pi exactly.
pub fn cvc_partner_angle(phi: f64, cfg: &SfConfig) -> f64 {
    let kf = fermi_surface_at(phi, cfg).0;
    let q = PI / (A_LCCO * ANG);
    let (cos, sin) = (phi.cos(), phi.sin());
    let px = kf * cos - q * sgn(cos);
    let py = kf * sin - q * sgn(sin);
    py.atan2(px).rem_euclid(2.0 * PI)
}

/// eps(phi) = eps0 sum_n exp(-1/2 (wrap(phi - phi_c,n) / sigma)^2) (Eq. 13)
/// on the centres from `cvc_centres`, which the caller computes once per call.
pub fn cvc_epsilon_profile(phi: f64, centres: &[f64; 8], cfg: &SfConfig) -> f64 {
    let mut out = 0.0;
    for &c in centres {
        let d = wrap(phi - c) / cfg.cvc_sigma;
        out += (-0.5 * (d * d)).exp();
    }
    cfg.cvc_epsilon * out
}

/// The x and y components of a field on the phi grid.
type Pair = [Vec<f64>; 2];

/// J* = v(phi) + eps(phi) v(phi') on the grid (Eq. 12 without the 1/(1-eps^2)
/// prefactor), in m/s.
fn cvc_currents(fs: &SfFermiSurface, centres: &[f64; 8], cfg: &SfConfig) -> Pair {
    let (jx, jy): (Vec<f64>, Vec<f64>) = fs
        .phi
        .iter()
        .zip(fs.vx.iter().zip(&fs.vy))
        .map(|(&phi, (&vx, &vy))| {
            let eps = cvc_epsilon_profile(phi, centres, cfg);
            let (_, _, _, pvx, pvy) = fermi_surface_at(cvc_partner_angle(phi, cfg), cfg);
            (vx + eps * pvx, vy + eps * pvy)
        })
        .unzip();
    [jx, jy]
}

/// The dressed current (J*_x, J*_y) on the grid in m/s.
pub fn cvc_velocity(cfg: &SfConfig) -> Result<(Vec<f64>, Vec<f64>), String> {
    let centres = cvc_centres(cfg)?;
    let [jx, jy] = cvc_currents(&fermi_surface_grid(cfg), &centres, cfg);
    Ok((jx, jy))
}

// ---------------------------------------------------------------------------
// Part A: Chambers integrator (SF-spec §5)
// ---------------------------------------------------------------------------

/// Spectral derivative on the periodic phi grid (SF-spec §5.2): FFT, multiply
/// by i kappa (kappa = 0 at k = 0 and N/2), inverse FFT, Re / N.
struct SpectralDerivative {
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
}

impl SpectralDerivative {
    fn new(n: usize) -> Self {
        let mut planner = FftPlanner::<f64>::new();
        Self {
            forward: planner.plan_fft_forward(n),
            inverse: planner.plan_fft_inverse(n),
        }
    }

    fn apply(&self, f: &[f64]) -> Vec<f64> {
        let n = f.len();
        let half = n / 2;
        let mut buf: Vec<Complex64> = f.iter().map(|&v| Complex64::new(v, 0.0)).collect();
        self.forward.process(&mut buf);
        for (k, z) in buf.iter_mut().enumerate() {
            let kappa = if k == 0 || k == half {
                0.0
            } else if k < half {
                k as f64
            } else {
                k as f64 - n as f64
            };
            *z = Complex64::new(-kappa * z.im, kappa * z.re);
        }
        self.inverse.process(&mut buf);
        let scale = n as f64;
        buf.iter().map(|z| z.re / scale).collect()
    }

    fn apply_pair(&self, f: &Pair) -> Pair {
        [self.apply(&f[0]), self.apply(&f[1])]
    }
}

/// E_n(R) = int_0^1 t^n e^(-R t) dt for n = 0..6 (SF-spec §5.3): a 28-term
/// series below R = 2 (k outer, n inner), the upward recursion above.
fn exp_moments(r: f64) -> [f64; 7] {
    let mut out = [0.0; 7];
    if r < 2.0 {
        let mut term = 1.0;
        for k in 0..28 {
            for (n, acc) in out.iter_mut().enumerate() {
                *acc += term / ((n + k + 1) as f64);
            }
            term = term * (-r) / ((k + 1) as f64);
        }
    } else {
        let em = (-r).exp();
        let mut e = -(-r).exp_m1() / r;
        out[0] = e;
        for (n, slot) in out.iter_mut().enumerate().skip(1) {
            e = (n as f64 * e - em) / r;
            *slot = e;
        }
    }
    out
}

/// M_s of L = f + M on a forward-ordered orbit (SF-spec §5.1): exact cells
/// with cubic-Hermite f, linear r, then the canonical two-pass closure of
/// M_s = I_s + A_s M_{s+1} (indices mod N). `f`, `fp` and `r` are in scan
/// order; the result is too.
fn orbit_corrections(f: [&[f64]; 2], fp: [&[f64]; 2], r: &[f64], h: f64) -> Pair {
    let n = r.len();
    let mut decay = vec![0.0; n];
    let mut cells = [vec![0.0; n], vec![0.0; n]];
    for s in 0..n {
        let s1 = if s + 1 == n { 0 } else { s + 1 };
        let (r_s, r_next) = (r[s], r[s1]);
        let k = 0.5 * (r_next - r_s) * h;
        decay[s] = (-0.5 * (r_s + r_next) * h).exp();
        let e = exp_moments(r_s * h);
        let k2 = 0.5 * k * k;
        let q0 = e[0] - k * e[2] + k2 * e[4];
        let q1 = e[1] - k * e[3] + k2 * e[5];
        let q2 = e[2] - k * e[4] + k2 * e[6];
        for (cell, (fc, fpc)) in cells.iter_mut().zip(f.iter().zip(&fp)) {
            let d = (fc[s1] - fc[s]) / h;
            let c1 = 6.0 * d - 4.0 * fpc[s] - 2.0 * fpc[s1];
            let c2 = 3.0 * fpc[s] + 3.0 * fpc[s1] - 6.0 * d;
            cell[s] = h * (fpc[s] * q0 + c1 * q1 + c2 * q2);
        }
    }
    let rtot = h * r.iter().sum::<f64>();
    let closure = -(-rtot).exp_m1();
    cells.map(|cell| {
        let mut acc = 0.0;
        let mut prod = 1.0;
        for (&i_s, &a_s) in cell.iter().zip(&decay) {
            acc += prod * i_s;
            prod *= a_s;
        }
        let mut m = vec![0.0; n];
        let mut next = acc / closure;
        m[0] = next;
        for ((m_s, &i_s), &a_s) in m.iter_mut().zip(&cell).zip(&decay).skip(1).rev() {
            next = i_s + a_s * next;
            *m_s = next;
        }
        m
    })
}

/// Gamma(phi) and f = J/Gamma at one (g, T), with f′ cached per sign of B
/// (f′ depends only on f and the scan order, so every field of that sign
/// reuses it).
struct Kernel {
    /// Gamma in s^-1, grid order.
    gamma: Vec<f64>,
    /// J/Gamma, grid order (= the B > 0 scan order).
    f: Pair,
    /// f′ in the B > 0 scan order.
    forward: Option<Pair>,
    /// (f, f′) in the B < 0 scan order idx_s = (-s) mod N.
    backward: Option<(Pair, Pair)>,
}

/// Per-config arrays shared by every transport evaluation. W(phi) and the
/// hot-spot sum depend on neither T nor g(x), so they are computed once; the
/// arithmetic per point is the same as `scattering_rate`'s.
struct Transport {
    cfg: SfConfig,
    fs: SfFermiSurface,
    /// W(phi_m).
    w: Vec<f64>,
    /// sum_n exp(-(wrap(phi_m - phi_n)/w)^2) when hot spots are on.
    hotspot: Option<Vec<f64>>,
    /// e^2 / (2 pi^2 hbar d).
    pref: f64,
    /// 2 pi / N.
    h: f64,
    /// weight * v_i.
    wv: Pair,
    /// weight * v_x * v_x.
    wvx2: Vec<f64>,
    deriv: SpectralDerivative,
}

impl Transport {
    /// `cfg` must already be validated (its doping is not used: g is an
    /// argument of every evaluation).
    fn new(cfg: &SfConfig) -> Self {
        let fs = fermi_surface_grid(cfg);
        let n = cfg.n_phi;
        let w = fs.phi.iter().map(|&p| inelastic_weight(p, cfg)).collect();
        let hotspot = hotspots_on(cfg).then(|| {
            let centres = hotspot_angles(cfg);
            fs.phi
                .iter()
                .map(|&p| gaussian_sum(p, &centres, cfg.hotspot_width))
                .collect()
        });
        let pref = E_CHARGE * E_CHARGE / (2.0 * PI * PI * HBAR_J_S * cfg.layer_spacing * ANG);
        let product =
            |a: &[f64], b: &[f64]| -> Vec<f64> { a.iter().zip(b).map(|(x, y)| x * y).collect() };
        let wv = [product(&fs.weight, &fs.vx), product(&fs.weight, &fs.vy)];
        let wvx2 = product(&wv[0], &fs.vx);
        Self {
            cfg: *cfg,
            fs,
            w,
            hotspot,
            pref,
            h: 2.0 * PI / n as f64,
            wv,
            wvx2,
            deriv: SpectralDerivative::new(n),
        }
    }

    /// Gamma(phi_m) in ps^-1 at doping strength g and temperature t.
    fn gamma_ps(&self, g: f64, t: f64) -> Vec<f64> {
        self.w
            .iter()
            .enumerate()
            .map(|(m, &w)| {
                let hs = self.hotspot.as_ref().map(|hs| hs[m]);
                gamma_from_parts(&self.cfg, g, t, w, hs)
            })
            .collect()
    }

    fn kernel(&self, g: f64, t: f64, forward: bool, backward: bool) -> Kernel {
        let gamma: Vec<f64> = self.gamma_ps(g, t).iter().map(|&v| v * PS_INV).collect();
        let ratio = |v: &[f64]| -> Vec<f64> { v.iter().zip(&gamma).map(|(a, b)| a / b).collect() };
        let f = [ratio(&self.fs.vx), ratio(&self.fs.vy)];
        let forward = forward.then(|| self.deriv.apply_pair(&f));
        let backward = backward.then(|| {
            let n = f[0].len();
            let reorder = |v: &[f64]| -> Vec<f64> { (0..n).map(|s| v[(n - s) % n]).collect() };
            let fb = [reorder(&f[0]), reorder(&f[1])];
            let fpb = self.deriv.apply_pair(&fb);
            (fb, fpb)
        });
        Kernel {
            gamma,
            f,
            forward,
            backward,
        }
    }

    /// sigma_ij(B) in S/m (i: outgoing velocity, j: the vector mean free
    /// path); B = 0 is the closed form L = J/Gamma.
    fn sigma(&self, k: &Kernel, b: f64) -> [[f64; 2]; 2] {
        if b == 0.0 {
            return self.sums([&k.f[0], &k.f[1]]);
        }
        let n = k.gamma.len();
        let babs = b.abs();
        let forward = b > 0.0;
        let scan = |s: usize| if forward { s } else { (n - s) % n };
        let r: Vec<f64> = (0..n)
            .map(|s| {
                let i = scan(s);
                k.gamma[i] / (self.fs.omega_per_tesla[i] * babs)
            })
            .collect();
        // f′ is cached on the kernel; recompute only if a caller did not ask
        // for this sign.
        let built;
        let (fo, fpo) = if forward {
            match &k.forward {
                Some(fp) => (&k.f, fp),
                None => {
                    built = (k.f.clone(), self.deriv.apply_pair(&k.f));
                    (&built.0, &built.1)
                }
            }
        } else {
            match &k.backward {
                Some((fb, fpb)) => (fb, fpb),
                None => {
                    let reorder =
                        |v: &[f64]| -> Vec<f64> { (0..n).map(|s| v[(n - s) % n]).collect() };
                    let fb = [reorder(&k.f[0]), reorder(&k.f[1])];
                    let fpb = self.deriv.apply_pair(&fb);
                    built = (fb, fpb);
                    (&built.0, &built.1)
                }
            }
        };
        let m = orbit_corrections([&fo[0], &fo[1]], [&fpo[0], &fpo[1]], &r, self.h);
        let mut l = [vec![0.0; n], vec![0.0; n]];
        for ((lc, fc), mc) in l.iter_mut().zip(fo).zip(&m) {
            for (s, (&f_s, &m_s)) in fc.iter().zip(mc).enumerate() {
                lc[scan(s)] = f_s + m_s;
            }
        }
        self.sums([&l[0], &l[1]])
    }

    /// pref h sum_m (w v_i)_m L_j,m in grid order, summed sequentially.
    fn sums(&self, l: [&[f64]; 2]) -> [[f64; 2]; 2] {
        let mut out = [[0.0; 2]; 2];
        for (row, wv) in out.iter_mut().zip(&self.wv) {
            for (slot, lj) in row.iter_mut().zip(l) {
                *slot = self.pref * self.h * wv.iter().zip(lj).map(|(a, b)| a * b).sum::<f64>();
            }
        }
        out
    }

    /// Closed-form rho(0, T) = 1 / sigma_xx in micro-Ohm cm.
    fn rho_zero(&self, g: f64, t: f64) -> f64 {
        let sum: f64 = self
            .wvx2
            .iter()
            .zip(self.gamma_ps(g, t))
            .map(|(&a, gam)| a / (gam * PS_INV))
            .sum();
        OHM_M_TO_MUOHM_CM / (self.pref * self.h * sum)
    }

    /// Jones-Zener R_H in m^3/C with the currents `j` (bare v or J*).
    fn hall(&self, g: f64, t: f64, j: &Pair) -> f64 {
        let gamma: Vec<f64> = self.gamma_ps(g, t).iter().map(|&v| v * PS_INV).collect();
        let diagonal = |wv: &[f64], jc: &[f64]| -> f64 {
            wv.iter()
                .zip(jc)
                .zip(&gamma)
                .map(|((a, b), gm)| a * b / gm)
                .sum::<f64>()
        };
        let sxx = self.pref * self.h * diagonal(&self.wv[0], &j[0]);
        let syy = self.pref * self.h * diagonal(&self.wv[1], &j[1]);
        let fy: Vec<f64> = j[1].iter().zip(&gamma).map(|(a, b)| a / b).collect();
        let dfy = self.deriv.apply(&fy);
        let cross: f64 = j[0]
            .iter()
            .zip(&gamma)
            .zip(&dfy)
            .map(|((jx, gm), d)| jx / gm * d)
            .sum();
        let sxy_b = self.pref * (E_CHARGE / HBAR_J_S) * self.h * cross;
        sxy_b / (sxx * syy)
    }

    /// ⟨W⟩_FS, the DOS-weighted Fermi-surface average.
    fn dos_average_w(&self) -> f64 {
        let num: f64 = self.w.iter().zip(&self.fs.weight).map(|(a, b)| a * b).sum();
        num / self.fs.weight.iter().sum::<f64>()
    }

    fn lambda_tr(&self, g: f64) -> f64 {
        HBAR_J_S * g * self.cfg.alpha1 * PS_INV * self.dos_average_w() / (2.0 * PI * KB_J_K)
    }

    /// rho_D = 1 / (pref h sum_m w v_x^2) in Ohm m s.
    fn drude_resistivity_per_rate(&self) -> f64 {
        1.0 / (self.pref * self.h * self.wvx2.iter().sum::<f64>())
    }

    fn resistivity_fit(
        &self,
        g: f64,
        t_min: f64,
        t_max: f64,
        step: f64,
    ) -> Result<SfResistivityFit, String> {
        let temps = fit_temperatures(t_min, t_max, step)?;
        let rho: Vec<f64> = temps.iter().map(|&t| self.rho_zero(g, t)).collect();
        fit_resistivity_polynomial(&temps, &rho)
    }

    /// lambda_rho = hbar (A1 / 1e8) / (2 pi k_B rho_D), A1 from the 15-60 K fit.
    fn lambda_resistive(&self, g: f64, rho_d: f64) -> Result<f64, String> {
        let (t_min, t_max) = SF_RHO_FIT_RANGE;
        let fit = self.resistivity_fit(g, t_min, t_max, 1.0)?;
        Ok(HBAR_J_S * (fit.a1 / OHM_M_TO_MUOHM_CM) / (2.0 * PI * KB_J_K * rho_d))
    }
}

/// [rho_xx, rho_yx] in micro-Ohm cm from the 2x2 conductivity.
fn rho_from_sigma(s: &[[f64; 2]; 2]) -> [f64; 2] {
    let det = s[0][0] * s[1][1] - s[0][1] * s[1][0];
    [
        s[1][1] / det * OHM_M_TO_MUOHM_CM,
        -s[1][0] / det * OHM_M_TO_MUOHM_CM,
    ]
}

fn fit_temperatures(t_min: f64, t_max: f64, step: f64) -> Result<Vec<f64>, String> {
    let ok =
        finite_non_negative(t_min) && t_max.is_finite() && t_max > t_min && finite_positive(step);
    if !ok {
        return Err(format!(
            "fit range needs 0 <= t_min < t_max and step > 0, got {t_min}, {t_max}, {step}"
        ));
    }
    let count = ((t_max - t_min) / step + 1e-9).floor();
    if count >= MAX_GRID_POINTS as f64 {
        return Err(format!("fit grid too fine: {count} steps"));
    }
    Ok((0..=count as usize)
        .map(|k| t_min + step * k as f64)
        .collect())
}

// ---------------------------------------------------------------------------
// Part A: conductivity, resistivity and magnetoresistance
// ---------------------------------------------------------------------------

/// sigma_ij(B) in S/m for each field (bare vertices; no CVCs in rho or MR).
/// `out[k][i][j]`: i = outgoing velocity, j = vector mean free path.
pub fn conductivity_tensor(
    cfg: &SfConfig,
    temperature: f64,
    fields: &[f64],
) -> Result<Vec<[[f64; 2]; 2]>, String> {
    validate(cfg)?;
    check_temperature(temperature)?;
    check_fields(fields)?;
    let g = g_of_x(cfg.doping)?;
    let transport = Transport::new(cfg);
    let kernel = transport.kernel(
        g,
        temperature,
        fields.iter().any(|&b| b > 0.0),
        fields.iter().any(|&b| b < 0.0),
    );
    Ok(fields
        .par_iter()
        .map(|&b| transport.sigma(&kernel, b))
        .collect())
}

/// [rho_xx, rho_yx] in micro-Ohm cm for each field.
pub fn resistivity_tensor(
    cfg: &SfConfig,
    temperature: f64,
    fields: &[f64],
) -> Result<Vec<[f64; 2]>, String> {
    Ok(conductivity_tensor(cfg, temperature, fields)?
        .iter()
        .map(rho_from_sigma)
        .collect())
}

/// Closed-form rho(0, T) in micro-Ohm cm.
pub fn zero_field_resistivity(cfg: &SfConfig, temperatures: &[f64]) -> Result<Vec<f64>, String> {
    validate(cfg)?;
    check_temperatures(temperatures)?;
    let g = g_of_x(cfg.doping)?;
    let transport = Transport::new(cfg);
    Ok(temperatures
        .iter()
        .map(|&t| transport.rho_zero(g, t))
        .collect())
}

/// 0 ..= field_max in `step` steps, inclusive; field_max is snapped down to a
/// step multiple (71 points at 35 T).
pub fn field_grid(field_max: f64, step: f64) -> Result<Vec<f64>, String> {
    if !finite_positive(field_max) || !finite_positive(step) {
        return Err(format!(
            "field_max and step must be finite and > 0, got {field_max} and {step}"
        ));
    }
    let count = (field_max / step + 1e-9).floor();
    if count >= MAX_GRID_POINTS as f64 {
        return Err(format!("field grid too fine: {count} steps"));
    }
    Ok((0..=count as usize).map(|k| step * k as f64).collect())
}

/// d rho / d mu0H on a uniform grid: central differences inside,
/// second-order one-sided at the ends (= np.gradient(..., edge_order=2)).
/// Fewer than three points fall back to a first-order slope.
pub fn field_derivative(rho: &[f64], step: f64) -> Vec<f64> {
    let n = rho.len();
    match n {
        0 => return Vec::new(),
        1 => return vec![0.0],
        2 => {
            let d = (rho[1] - rho[0]) / step;
            return vec![d, d];
        }
        _ => {}
    }
    let mut out = vec![0.0; n];
    out[0] = (-3.0 * rho[0] + 4.0 * rho[1] - rho[2]) / (2.0 * step);
    for (slot, w) in out[1..n - 1].iter_mut().zip(rho.windows(3)) {
        *slot = (w[2] - w[0]) / (2.0 * step);
    }
    out[n - 1] = (3.0 * rho[n - 1] - 4.0 * rho[n - 2] + rho[n - 3]) / (2.0 * step);
    out
}

/// The MR family on 0 ..= field_max: rayon over (T, B) pairs, f′ once per T.
pub fn magnetoresistance(
    cfg: &SfConfig,
    temperatures: &[f64],
    field_max: f64,
    field_step: f64,
) -> Result<SfMagnetoresistance, String> {
    validate(cfg)?;
    check_temperatures(temperatures)?;
    let fields = field_grid(field_max, field_step)?;
    if fields.len() < 3 {
        return Err(format!(
            "the field grid needs at least 3 points (field_max {field_max}, step {field_step})"
        ));
    }
    let g = g_of_x(cfg.doping)?;
    let transport = Transport::new(cfg);
    let kernels: Vec<Kernel> = temperatures
        .par_iter()
        .map(|&t| transport.kernel(g, t, true, false))
        .collect();
    let nb = fields.len();
    let rho: Vec<[f64; 2]> = (0..temperatures.len() * nb)
        .into_par_iter()
        .map(|p| rho_from_sigma(&transport.sigma(&kernels[p / nb], fields[p % nb])))
        .collect();

    let mut out = SfMagnetoresistance {
        temperatures: temperatures.to_vec(),
        fields: fields.clone(),
        rho_xx: Vec::with_capacity(temperatures.len()),
        rho_yx: Vec::with_capacity(temperatures.len()),
        mr: Vec::with_capacity(temperatures.len()),
        drho_dh: Vec::with_capacity(temperatures.len()),
        h_over_t: Vec::with_capacity(temperatures.len()),
    };
    for (row, &t) in rho.chunks(nb).zip(temperatures) {
        let rho_xx: Vec<f64> = row.iter().map(|r| r[0]).collect();
        let rho0 = rho_xx[0];
        out.mr
            .push(rho_xx.iter().map(|&r| r / rho0 - 1.0).collect());
        out.drho_dh.push(field_derivative(&rho_xx, field_step));
        out.h_over_t.push(fields.iter().map(|&b| b / t).collect());
        out.rho_yx.push(row.iter().map(|r| r[1]).collect());
        out.rho_xx.push(rho_xx);
    }
    Ok(out)
}

/// gamma_1 = (rho(field + step) - rho(field - step)) / (2 step), micro-Ohm cm/T.
pub fn high_field_slope(
    cfg: &SfConfig,
    temperature: f64,
    field: f64,
    step: f64,
) -> Result<f64, String> {
    if !finite_positive(step) {
        return Err(format!("step must be finite and > 0, got {step}"));
    }
    let r = resistivity_tensor(cfg, temperature, &[field - step, field + step])?;
    Ok((r[1][0] - r[0][0]) / (2.0 * step))
}

/// rho0 + A1 T + A2 T^2 by least squares: sequential power sums, 3x3 normal
/// equations solved by Cramer's rule (SF-spec §3.4).
pub fn fit_resistivity_polynomial(
    temperatures: &[f64],
    rho: &[f64],
) -> Result<SfResistivityFit, String> {
    if temperatures.len() != rho.len() || temperatures.len() < 3 {
        return Err(format!(
            "need at least 3 (T, rho) pairs of equal length, got {} and {}",
            temperatures.len(),
            rho.len()
        ));
    }
    if !temperatures.iter().chain(rho).all(|v| v.is_finite()) {
        return Err("temperatures and rho must be finite".to_string());
    }
    let mut s = [0.0; 5];
    let mut r = [0.0; 3];
    for (&t, &v) in temperatures.iter().zip(rho) {
        let mut p = 1.0;
        for (k, sk) in s.iter_mut().enumerate() {
            *sk += p;
            if k < 3 {
                r[k] += v * p;
            }
            p *= t;
        }
    }
    let det3 = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let normal = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]];
    let det = det3(&normal);
    if det == 0.0 || !det.is_finite() {
        return Err("singular normal equations (temperatures must be distinct)".to_string());
    }
    let solve = |c: usize| {
        let mut mc = normal;
        for (row, &rr) in mc.iter_mut().zip(&r) {
            row[c] = rr;
        }
        det3(&mc) / det
    };
    let t_min = temperatures.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = temperatures
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    Ok(SfResistivityFit {
        rho0: solve(0),
        a1: solve(1),
        a2: solve(2),
        t_min,
        t_max,
    })
}

/// Fit of the model's own closed-form rho(0, T) on [t_min, t_max] at `step`
/// (46 points for 15-60 K).
pub fn resistivity_fit(
    cfg: &SfConfig,
    t_min: f64,
    t_max: f64,
    step: f64,
) -> Result<SfResistivityFit, String> {
    validate(cfg)?;
    let g = g_of_x(cfg.doping)?;
    Transport::new(cfg).resistivity_fit(g, t_min, t_max, step)
}

// ---------------------------------------------------------------------------
// Part A: Hall coefficient (Jones-Zener, Eqs. 14-15)
// ---------------------------------------------------------------------------

/// R_H(T) in m^3/C, Jones-Zener with (cvc) or without CVCs; J* and the CVC
/// centres are built once for the whole curve.
pub fn hall_coefficient_curve(
    cfg: &SfConfig,
    temperatures: &[f64],
    cvc: bool,
) -> Result<Vec<f64>, String> {
    validate(cfg)?;
    check_temperatures(temperatures)?;
    let g = g_of_x(cfg.doping)?;
    let transport = Transport::new(cfg);
    let currents = if cvc {
        let centres = eight_angles(zone_boundary_crossing(cfg)?);
        cvc_currents(&transport.fs, &centres, cfg)
    } else {
        [transport.fs.vx.clone(), transport.fs.vy.clone()]
    };
    Ok(temperatures
        .par_iter()
        .map(|&t| transport.hall(g, t, &currents))
        .collect())
}

/// Jones-Zener R_H in m^3/C; CVCs per the printed Eqs. 14-15 (one dressed
/// vertex in sigma_xx, both in sigma_xy), centred on the FS x ZB crossing.
pub fn hall_coefficient(cfg: &SfConfig, temperature: f64, cvc: bool) -> Result<f64, String> {
    Ok(hall_coefficient_curve(cfg, &[temperature], cvc)?[0])
}

/// R_H(cfg, T, CVC) / R_H(case (a), CVC), with case (a) = no inelastic term and
/// no hot spots. The denominator is T-independent (a uniform Gamma cancels in
/// Jones-Zener); (d)/(a) on the info card against the paper's 4.3/8.7 = 0.49.
pub fn hall_ratio_to_case_a(cfg: &SfConfig, temperature: f64) -> Result<f64, String> {
    let case_a = SfConfig {
        inelastic: SfInelasticForm::None,
        hotspots: false,
        ..*cfg
    };
    Ok(hall_coefficient(cfg, temperature, true)? / hall_coefficient(&case_a, temperature, true)?)
}

/// Low-field Chambers R_H = rho_yx / B in m^3/C (no CVC), the check on
/// Jones-Zener.
pub fn hall_coefficient_chambers(
    cfg: &SfConfig,
    temperature: f64,
    field: f64,
) -> Result<f64, String> {
    if field == 0.0 || !field.is_finite() {
        return Err(format!("field must be finite and nonzero, got {field}"));
    }
    let r = resistivity_tensor(cfg, temperature, &[field])?;
    Ok(r[0][1] / OHM_M_TO_MUOHM_CM / field)
}

/// Drude 1/(n e) on the fixed x = 0.159 Fermi surface, n = Luttinger holes /
/// (a^2 d), in m^3/C.
pub fn drude_hall_coefficient(cfg: &SfConfig) -> Result<f64, String> {
    let holes = luttinger_holes(cfg)?;
    let a = A_LCCO * ANG;
    let n = holes / (a * a * cfg.layer_spacing * ANG);
    Ok(1.0 / (n * E_CHARGE))
}

/// Nominal Drude a^2 d / ((1 - x) e) in m^3/C: 1/(n e) with n = 1 - x holes
/// per Cu.
pub fn drude_hall_coefficient_nominal(cfg: &SfConfig) -> Result<f64, String> {
    validate(cfg)?;
    let a = A_LCCO * ANG;
    let n = (1.0 - cfg.doping) / (a * a * cfg.layer_spacing * ANG);
    Ok(1.0 / (n * E_CHARGE))
}

// ---------------------------------------------------------------------------
// Part A: couplings and the doping series
// ---------------------------------------------------------------------------

/// λ_tr, the Fermi-surface-averaged T-linear coupling (Allen's form applied to
/// the microscopic rate): hbar g(x) alpha1 ⟨W⟩_FS / (2 pi k_B), DOS-weighted
/// (0.41758274 g for the paper's form). It is not Allen's resistivity-derived
/// lambda; that is `lambda_resistive`.
pub fn lambda_tr(cfg: &SfConfig) -> Result<f64, String> {
    validate(cfg)?;
    let g = g_of_x(cfg.doping)?;
    Ok(Transport::new(cfg).lambda_tr(g))
}

/// rho_D: the model's zero-field rho per unit isotropic rate on the same FS,
/// in Ohm m s (rho = rho_D Gamma exactly when Gamma is uniform).
pub fn drude_resistivity_per_rate(cfg: &SfConfig) -> Result<f64, String> {
    validate(cfg)?;
    Ok(Transport::new(cfg).drude_resistivity_per_rate())
}

/// λ_ρ, the Allen-Drude coupling from the model's own resistivity (D4):
/// hbar A1 / (2 pi k_B rho_D), with A1 from the 15-60 K fit.
pub fn lambda_resistive(cfg: &SfConfig) -> Result<f64, String> {
    validate(cfg)?;
    let g = g_of_x(cfg.doping)?;
    let transport = Transport::new(cfg);
    transport.lambda_resistive(g, transport.drude_resistivity_per_rate())
}

/// g, Tc, lambda_tr, lambda_rho, rho(0, T) and MR(field, T) at each tabulated
/// x: rayon over (x, B) pairs. The config's own doping is validated but not
/// used; the Fermi surface is the x = 0.159 one at every x (only g varies).
pub fn doping_sweep(
    cfg: &SfConfig,
    temperature: f64,
    field: f64,
) -> Result<Vec<SfDopingPoint>, String> {
    validate(cfg)?;
    check_temperature(temperature)?;
    check_fields(&[field])?;
    let transport = Transport::new(cfg);
    let gs: Vec<f64> = G_TABLE_LCCO.iter().map(|&(_, g)| g).collect();
    let kernels: Vec<Kernel> = gs
        .par_iter()
        .map(|&g| transport.kernel(g, temperature, field > 0.0, field < 0.0))
        .collect();
    let pair_fields = [0.0, field];
    let rho: Vec<f64> = (0..2 * gs.len())
        .into_par_iter()
        .map(|p| rho_from_sigma(&transport.sigma(&kernels[p / 2], pair_fields[p % 2]))[0])
        .collect();
    let rho_d = transport.drude_resistivity_per_rate();
    let lambda_rho: Vec<f64> = gs
        .par_iter()
        .map(|&g| transport.lambda_resistive(g, rho_d))
        .collect::<Result<_, _>>()?;
    Ok(G_TABLE_LCCO
        .iter()
        .zip(rho.chunks(2))
        .zip(lambda_rho)
        .map(|((&(x, g), r), lambda_rho)| SfDopingPoint {
            doping: x,
            g,
            tc: tc_of_x(x),
            lambda_tr: transport.lambda_tr(g),
            lambda_rho,
            rho0: r[0],
            mr: r[1] / r[0] - 1.0,
        })
        .collect())
}

/// The LCCO doping whose lambda_tr (paper form, default FS) equals `lam`, by
/// inverse linear interpolation of G_TABLE_LCCO (the first crossing in
/// ascending x); None outside [lambda_tr(0.188), lambda_tr(0.116)].
pub fn equivalent_doping(lam: f64) -> Option<f64> {
    let per_g = lambda_tr(&SfConfig::default()).ok()?;
    let g = lam / per_g;
    for pair in G_TABLE_LCCO.windows(2) {
        let ((x0, g0), (x1, g1)) = (pair[0], pair[1]);
        if (g1..=g0).contains(&g) {
            return Some(if g0 == g1 {
                x0
            } else {
                x0 + (x1 - x0) * (g0 - g) / (g0 - g1)
            });
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Part B — pairing map (SPECULATIVE)
// ---------------------------------------------------------------------------

/// **SPECULATIVE** Tc(lambda) in K; 0 at and below the threshold.
pub fn tc_from_lambda(lam: f64, p: &SfPairingMap) -> f64 {
    match p.form {
        SfPairingForm::PowerLaw => {
            let x = (lam - p.lambda_c) / (p.lambda_ref - p.lambda_c);
            // np.clip(x, 0, None): NaN stays NaN.
            let x = if x < 0.0 { 0.0 } else { x };
            p.tc_ref * x.powf(p.exponent)
        }
        SfPairingForm::McMillan => {
            let den = lam - p.mu_star * (1.0 + 0.62 * lam);
            if den > 0.0 {
                (p.omega_k / 1.45) * (-1.04 * (1.0 + lam) / den).exp()
            } else {
                0.0
            }
        }
    }
}

fn lambda_from_tc_unchecked(tc: f64, p: &SfPairingMap) -> f64 {
    match p.form {
        SfPairingForm::PowerLaw => {
            p.lambda_c + (p.lambda_ref - p.lambda_c) * (tc / p.tc_ref).powf(1.0 / p.exponent)
        }
        SfPairingForm::McMillan => {
            let l = -(1.45 * tc / p.omega_k).ln() / 1.04;
            (1.0 + l * p.mu_star) / (l * (1.0 - 0.62 * p.mu_star) - 1.0)
        }
    }
}

/// **SPECULATIVE** inverse map; Err unless 0 < tc < tc_ceiling(p), and Err
/// where the inverse rounds onto the threshold so Tc(lambda) = 0 (default power
/// law: tc <= 6.173e-6 K); the glue quantities divide by Tc(lambda_bar).
pub fn lambda_from_tc(tc: f64, p: &SfPairingMap) -> Result<f64, String> {
    validate_pairing(p)?;
    check_tc_target(tc, p)?;
    Ok(lambda_from_tc_unchecked(tc, p))
}

/// **SPECULATIVE** lambda at which Tc vanishes.
pub fn lambda_threshold(p: &SfPairingMap) -> f64 {
    match p.form {
        SfPairingForm::PowerLaw => p.lambda_c,
        SfPairingForm::McMillan => p.mu_star / (1.0 - 0.62 * p.mu_star),
    }
}

/// **SPECULATIVE** supremum of Tc(lambda) in K: infinite for the power law,
/// (omega/1.45) exp(-1.04/(1 - 0.62 mu*)) for McMillan.
pub fn tc_ceiling(p: &SfPairingMap) -> f64 {
    match p.form {
        SfPairingForm::PowerLaw => f64::INFINITY,
        SfPairingForm::McMillan => (p.omega_k / 1.45) * (-1.04 / (1.0 - 0.62 * p.mu_star)).exp(),
    }
}

/// **SPECULATIVE** S = d ln Tc / d ln lambda.
pub fn log_sensitivity(lam: f64, p: &SfPairingMap) -> f64 {
    match p.form {
        SfPairingForm::PowerLaw => p.exponent * lam / (lam - p.lambda_c),
        SfPairingForm::McMillan => {
            let d = lam - p.mu_star * (1.0 + 0.62 * lam);
            1.04 * lam * (1.0 + 0.38 * p.mu_star) / (d * d)
        }
    }
}

/// lambda_tr per unit g on the default (paper-form, 1024-point) Fermi surface.
fn lambda_tr_per_g() -> f64 {
    lambda_tr(&SfConfig::default()).unwrap_or(f64::NAN)
}

/// **SPECULATIVE** calibration data: (x, Tc, lambda_tr(x)) at the Fig. 5c
/// labels, lambda = (lambda_tr per unit g) x g(x) as in the power-law fit.
pub fn calibration_points() -> [(f64, f64, f64); 4] {
    let per_g = lambda_tr_per_g();
    TC_LABELS_FIG5C.map(|(x, tc)| (x, tc, per_g * g_of_x(x).unwrap_or(f64::NAN)))
}

/// **SPECULATIVE** rms (K) of Tc(lambda) against the four Fig. 5c labels.
pub fn pairing_fit_rms(p: &SfPairingMap) -> f64 {
    let points = calibration_points();
    let sum: f64 = points
        .iter()
        .map(|&(_, tc, lam)| {
            let d = tc_from_lambda(lam, p) - tc;
            d * d
        })
        .sum();
    (sum / points.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Part C — moire glue what-if (SPECULATIVE)
// ---------------------------------------------------------------------------

/// **SPECULATIVE** glue profile m~ = -cos(pi m), m in [0, 1] (the profile of
/// `gap_modulation` with its default phase shift pi).
pub fn moire_profile(m: f64) -> f64 {
    -(PI * m).cos()
}

fn coherence_length_unchecked(glue: &SfGlueConfig) -> f64 {
    let ratio = TC_FETE_FILM / glue.tc_target;
    match glue.coherence {
        SfCoherence::Dirty => SF_COHERENCE_LENGTH_FILM * ratio.sqrt(),
        SfCoherence::Clean | SfCoherence::Local => SF_COHERENCE_LENGTH_FILM * ratio,
    }
}

fn coherence_factor_unchecked(glue: &SfGlueConfig) -> f64 {
    match glue.coherence {
        SfCoherence::Local => 1.0,
        SfCoherence::Clean | SfCoherence::Dirty => {
            (-(coherence_length_unchecked(glue) / glue.moire_period)).exp()
        }
    }
}

fn glue_gap_scale_unchecked(glue: &SfGlueConfig) -> f64 {
    glue.delta_avg * (glue.tc_target / TC_FETE_FILM)
}

/// **SPECULATIVE** xi(Tc*) = xi_film (TC_FETE_FILM / Tc*)^s in Angstrom: s = 1
/// clean (also reported in the local limit), s = 1/2 dirty.
pub fn coherence_length(glue: &SfGlueConfig) -> Result<f64, String> {
    validate_glue(glue)?;
    Ok(coherence_length_unchecked(glue))
}

/// **SPECULATIVE** F = exp(-(xi / L_m)), evaluated directly (Rust has no
/// cpdm_amplitude; Python's clamps xi/L_m at 100); 1 in the local limit.
pub fn coherence_factor(glue: &SfGlueConfig) -> Result<f64, String> {
    validate_glue(glue)?;
    Ok(coherence_factor_unchecked(glue))
}

/// **SPECULATIVE** Delta_bar = delta_avg (Tc* / TC_FETE_FILM) in meV (D2: the
/// film's 2 Delta / k_B Tc at every Tc*).
pub fn glue_gap_scale(glue: &SfGlueConfig) -> Result<f64, String> {
    validate_glue(glue)?;
    Ok(glue_gap_scale_unchecked(glue))
}

/// **SPECULATIVE** glue-modulated gap on a moire pattern m in [0, 1]:
/// lambda(r) = lambda_bar (1 + eta m~(r)), local Tc(r), Delta_loc = Delta_bar
/// Tc(r) / Tc(lambda_bar), dressed Delta = Delta_bar + F (Delta_loc -
/// Delta_bar). Tc(r) = 0 regions exist only in the local limit: F < 1 keeps
/// Delta >= Delta_bar (1 - F) > 0.
pub fn glue_modulated_gap(pattern: &[f64], glue: &SfGlueConfig) -> Result<SfGlueResult, String> {
    validate_glue(glue)?;
    if pattern.is_empty() {
        return Err("pattern must not be empty".to_string());
    }
    if !pattern.iter().all(|m| m.is_finite()) {
        return Err("pattern must be finite".to_string());
    }
    let p = &glue.pairing;
    let lambda_bar = lambda_from_tc_unchecked(glue.tc_target, p);
    let tc_bar = tc_from_lambda(lambda_bar, p);
    let delta_bar = glue_gap_scale_unchecked(glue);
    let factor = coherence_factor_unchecked(glue);
    let samples: Vec<[f64; 4]> = pattern
        .par_iter()
        .map(|&m| {
            let lam = lambda_bar * (1.0 + glue.eta * moire_profile(m));
            let tc = tc_from_lambda(lam, p);
            let local = delta_bar * tc / tc_bar;
            [lam, tc, local, delta_bar + factor * (local - delta_bar)]
        })
        .collect();
    let span = |k: usize| {
        let (lo, hi) = samples
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| {
                (lo.min(s[k]), hi.max(s[k]))
            });
        hi - lo
    };
    let normal = samples.iter().filter(|s| s[1] == 0.0).count() as f64 / samples.len() as f64;
    Ok(SfGlueResult {
        lambda_bar,
        sensitivity: log_sensitivity(lambda_bar, p),
        coherence_length: coherence_length_unchecked(glue),
        coherence_factor: factor,
        moire_period: glue.moire_period,
        relative_modulation: span(3) / (2.0 * delta_bar),
        local_relative_modulation: span(2) / (2.0 * delta_bar),
        local_normal_fraction: normal,
        normal_fraction: if glue.coherence == SfCoherence::Local {
            normal
        } else {
            0.0
        },
        lambda_map: samples.iter().map(|s| s[0]).collect(),
        tc_map: samples.iter().map(|s| s[1]).collect(),
        gap: samples.iter().map(|s| s[3]).collect(),
    })
}

/// **SPECULATIVE** small-eta dressed amplitude in meV: F Delta_bar eta
/// S(lambda_bar). As eta -> 0 the glue map tends to Delta_bar - this x
/// cos(pi m), `gap_modulation`'s form.
pub fn linearized_glue_amplitude(glue: &SfGlueConfig) -> Result<f64, String> {
    validate_glue(glue)?;
    let lambda_bar = lambda_from_tc_unchecked(glue.tc_target, &glue.pairing);
    Ok(coherence_factor_unchecked(glue)
        * glue_gap_scale_unchecked(glue)
        * glue.eta
        * log_sensitivity(lambda_bar, &glue.pairing))
}

/// Amplitude-route reference DELTA_AMPLITUDE / DELTA_AVG = 0.165 ("amplitude
/// route, measured gaps (13.5 K)"): measured gaps, already coherence-dressed;
/// a constant, not a curve in Tc*.
pub fn bcs_route_relative_modulation() -> f64 {
    DELTA_AMPLITUDE / DELTA_AVG
}

fn local_imprint_unchecked(eta: f64, tc_target: f64, p: &SfPairingMap) -> (f64, f64) {
    let lb = lambda_from_tc_unchecked(tc_target, p);
    let tcb = tc_from_lambda(lb, p);
    let exact =
        (tc_from_lambda(lb * (1.0 + eta), p) - tc_from_lambda(lb * (1.0 - eta), p)) / (2.0 * tcb);
    (exact, eta * log_sensitivity(lb, p))
}

/// **SPECULATIVE** pattern-free local imprint (m spans [0, 1]):
/// ([Tc(lambda_bar(1+eta)) - Tc(lambda_bar(1-eta))] / (2 Tc(lambda_bar)),
/// eta S(lambda_bar)).
pub fn local_imprint(eta: f64, tc_target: f64, p: &SfPairingMap) -> Result<(f64, f64), String> {
    validate_pairing(p)?;
    check_eta(eta)?;
    check_tc_target(tc_target, p)?;
    Ok(local_imprint_unchecked(eta, tc_target, p))
}

/// (dressed exact, local exact, local linear) imprints against the Tc targets.
pub type SfImprintCurves = (Vec<f64>, Vec<f64>, Vec<f64>);

/// **SPECULATIVE** (dressed exact, local exact, local linear) against the Tc
/// targets, for `glue.coherence` and `glue.moire_period`.
pub fn imprint_curve(glue: &SfGlueConfig, tc_targets: &[f64]) -> Result<SfImprintCurves, String> {
    validate_glue(glue)?;
    let mut dressed = Vec::with_capacity(tc_targets.len());
    let mut local = Vec::with_capacity(tc_targets.len());
    let mut linear = Vec::with_capacity(tc_targets.len());
    for &tc_target in tc_targets {
        let gi = SfGlueConfig { tc_target, ..*glue };
        validate_glue(&gi)?;
        let (exact, lin) = local_imprint_unchecked(gi.eta, tc_target, &gi.pairing);
        dressed.push(coherence_factor_unchecked(&gi) * exact);
        local.push(exact);
        linear.push(lin);
    }
    Ok((dressed, local, linear))
}

/// **SPECULATIVE** local limit only: Tc(r) = 0 somewhere iff Tc* < Tc(lambda_c
/// / (1 - eta)); 0.0 when the map never fragments (McMillan with mu* = 0).
pub fn fragmentation_tc(glue: &SfGlueConfig) -> Result<f64, String> {
    validate_glue(glue)?;
    Ok(tc_from_lambda(
        lambda_threshold(&glue.pairing) / (1.0 - glue.eta),
        &glue.pairing,
    ))
}

// ---------------------------------------------------------------------------
// Part D — hole-doped quadrature contrast
// ---------------------------------------------------------------------------

/// rho(H, T) = rho0 + A_g T + B T^2 + sqrt((alpha k_B T)^2 + (gamma mu_B mu0H)^2)
/// in micro-Ohm cm (Ayres et al. 2021; illustrative Tl2201 parameters).
pub fn quadrature_resistivity(temperature: f64, field: f64, q: &SfQuadratureConfig) -> f64 {
    let t = temperature;
    q.rho0
        + q.a_g * t
        + q.b * t * t
        + ((q.alpha_kb * t).powi(2) + (q.gamma_mub * field).powi(2)).sqrt()
}

/// d rho / d mu0H of the quadrature form in micro-Ohm cm/T; it collapses
/// against H/T by construction. 0 where the root vanishes (T = H = 0, or
/// both terms underflow), as Python's `np.divide(..., where=root > 0)`; a
/// NaN input still gives NaN (Python raises).
pub fn quadrature_field_derivative(temperature: f64, field: f64, q: &SfQuadratureConfig) -> f64 {
    let t = temperature;
    let root = ((q.alpha_kb * t).powi(2) + (q.gamma_mub * field).powi(2)).sqrt();
    if root == 0.0 {
        0.0
    } else {
        q.gamma_mub * q.gamma_mub * field / root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::by_name;
    use crate::pdw::moire_locked_period;

    /// Behaviour tests run at n_phi = 256 (SF-spec §9.2); the anchors and the
    /// I3 low-field checks use the default 1024.
    fn cfg256() -> SfConfig {
        SfConfig {
            n_phi: 256,
            ..Default::default()
        }
    }

    fn assert_close(v: f64, want: f64, tol: f64, what: &str) {
        assert!(
            (v - want).abs() <= tol * want.abs(),
            "{what}: got {v:e}, want {want:e} (rel {:.2e} > {tol:e})",
            ((v - want) / want).abs()
        );
    }

    /// MR(b) = rho_xx(b) / rho_xx(0) - 1 at temperature t.
    fn mr_at(cfg: &SfConfig, t: f64, b: f64) -> f64 {
        let r = resistivity_tensor(cfg, t, &[0.0, b]).unwrap();
        r[1][0] / r[0][0] - 1.0
    }

    fn rh(cfg: &SfConfig, t: f64, cvc: bool) -> f64 {
        hall_coefficient(cfg, t, cvc).unwrap()
    }

    fn strictly_increasing(v: &[f64]) -> bool {
        v.windows(2).all(|w| w[1] > w[0])
    }

    fn strictly_decreasing(v: &[f64]) -> bool {
        v.windows(2).all(|w| w[1] < w[0])
    }

    /// max / min of a positive series.
    fn spread(v: &[f64]) -> f64 {
        let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = v.iter().copied().fold(f64::INFINITY, f64::min);
        max / min
    }

    /// np.linspace(start, stop, num): k * step + start, the last point exact.
    fn linspace(start: f64, stop: f64, num: usize) -> Vec<f64> {
        let step = (stop - start) / (num - 1) as f64;
        let mut v: Vec<f64> = (0..num).map(|k| k as f64 * step + start).collect();
        v[num - 1] = stop;
        v
    }

    /// A moire-like pattern on [0, 1] with both extremes present.
    fn ramp_pattern(n: usize) -> Vec<f64> {
        (0..n).map(|k| k as f64 / (n - 1) as f64).collect()
    }

    fn case_a(cfg: &SfConfig) -> SfConfig {
        SfConfig {
            inelastic: SfInelasticForm::None,
            hotspots: false,
            ..*cfg
        }
    }

    fn mcmillan() -> SfPairingMap {
        SfPairingMap {
            form: SfPairingForm::McMillan,
            ..Default::default()
        }
    }

    // --- Tables -------------------------------------------------------------

    #[test]
    fn test_g_table_and_snap() {
        assert_eq!(DOPINGS_LCCO.len(), 19);
        assert_eq!(g_of_x(SF_DOPING_REF).unwrap(), 1.0);
        assert_eq!(g_of_x(0.116).unwrap(), 6.50);
        assert_eq!(g_of_x(0.188).unwrap(), 0.55);
        assert!(G_TABLE_LCCO
            .windows(2)
            .all(|w| w[1].0 > w[0].0 && w[1].1 <= w[0].1));
        // Exact at the nodes, linear between them.
        for (&(x, g), &d) in G_TABLE_LCCO.iter().zip(&DOPINGS_LCCO) {
            assert_eq!(x, d);
            assert_eq!(g_of_x(x).unwrap(), g);
        }
        assert_close(
            g_of_x(0.1175).unwrap(),
            0.5 * (6.50 + 5.75),
            1e-12,
            "midpoint",
        );
        for bad in [0.1, 0.2, f64::NAN] {
            assert!(g_of_x(bad).is_err(), "x = {bad}");
        }
        assert_eq!(snap_doping(0.15), 0.149);
        // 0.1195 is an exact floating-point tie between 0.119 and 0.120.
        assert_eq!(snap_doping(0.1195), 0.119);
        assert_eq!(snap_doping(f64::NAN), SF_DOPING_REF);
        assert_eq!(snap_doping(f64::INFINITY), SF_DOPING_REF);
        assert_eq!(snap_doping(0.0), 0.116);
        assert_eq!(snap_doping(0.5), 0.188);
        // Tc table: the inferred plateau, the labels and the dome edge.
        assert_eq!(tc_of_x(0.05), 25.0);
        assert_eq!(tc_of_x(0.116), 25.0);
        assert_eq!(tc_of_x(0.14), 15.0);
        assert_close(tc_of_x(0.159), 8.35, 1e-12, "Tc(0.159)");
        assert_eq!(tc_of_x(0.18), 0.0);
        assert_eq!(tc_of_x(0.188), 0.0);
        assert_eq!(film_of_x(0.159), "S1");
        assert_eq!(film_of_x(0.165), "S2");
    }

    #[test]
    fn test_doping_regimes() {
        assert_eq!(doping_regime(0.138), SfDopingRegime::FsReconstruction);
        assert_eq!(doping_regime(0.141), SfDopingRegime::Superconducting);
        assert_eq!(doping_regime(0.180), SfDopingRegime::BeyondDome);
        assert_eq!(doping_regime(0.188), SfDopingRegime::BeyondDome);
        let ids = ["fs_reconstruction", "superconducting", "beyond_dome"];
        for (regime, id) in SfDopingRegime::ALL.iter().zip(ids) {
            assert_eq!(regime.as_str(), id);
            assert!(!regime.label().is_empty() && !regime.blurb().is_empty());
        }
    }

    #[test]
    fn test_enum_labels_match_python() {
        let inelastic = ["afm_boundary", "isotropic", "antinodal", "none"];
        for (form, id) in SfInelasticForm::ALL.iter().zip(inelastic) {
            assert_eq!(form.as_str(), id);
        }
        assert_eq!(SfInelasticForm::Isotropic.label(), "Isotropic (Fig. 9c)");
        assert_eq!(
            SfInelasticForm::Antinodal.label(),
            "Antinodal (Fig. 9e, 13c)"
        );
        assert_eq!(
            SfPairingForm::McMillan.label(),
            "McMillan form, best fit — fails"
        );
        let coherence = [
            ("clean", "Clean: ξ ∝ 1/Tc (default)"),
            ("dirty", "Dirty: ξ ∝ Tc^−½"),
            ("local", "Local limit (no coherence)"),
        ];
        for (model, (id, label)) in SfCoherence::ALL.iter().zip(coherence) {
            assert_eq!(model.as_str(), id);
            assert_eq!(model.label(), label);
        }
        assert_eq!(SfCoherence::default(), SfCoherence::Clean);
        assert_eq!(SfInelasticForm::default(), SfInelasticForm::AfmBoundary);
        assert_eq!(SfPairingForm::default(), SfPairingForm::PowerLaw);
    }

    // --- Fermi surface and CVC geometry --------------------------------------

    #[test]
    fn test_fermi_surface_extrema_and_normal_velocity() {
        let c = SfConfig::default();
        assert_eq!(fermi_surface_at(0.0, &c).0, KF0_LCCO);
        assert_close(
            fermi_surface_at(PI / 4.0, &c).0,
            KF0_LCCO * (1.0 + KF_ANISOTROPY_LCCO),
            1e-14,
            "k_F(node)",
        );
        let fs = fermi_surface(&c).unwrap();
        assert_eq!(fs.phi.len(), SF_N_PHI);
        assert_eq!(spread(&fs.kf), 1.0 + KF_ANISOTROPY_LCCO);
        for (m, &phi) in fs.phi.iter().enumerate() {
            // Tangent of k(phi) = k_F(phi)(cos, sin), with dk_F/dphi = 2 k_F0 alpha sin 4 phi.
            let dkf = 2.0 * c.kf0 * c.kf_anisotropy * (4.0 * phi).sin();
            let tx = dkf * phi.cos() - fs.kf[m] * phi.sin();
            let ty = dkf * phi.sin() + fs.kf[m] * phi.cos();
            let dot = fs.vx[m] * tx + fs.vy[m] * ty;
            assert!(
                dot.abs() <= 1e-12 * fs.vf[m] * tx.hypot(ty),
                "m = {m}: {dot:e}"
            );
        }
    }

    #[test]
    fn test_luttinger_count() {
        let holes = luttinger_holes(&SfConfig::default()).unwrap();
        assert_close(holes, 0.8391564248900439, 1e-12, "Luttinger");
        assert!((holes - 0.841).abs() < 0.005);
    }

    #[test]
    fn test_zone_boundary_crossing() {
        let c = SfConfig::default();
        let deg = zone_boundary_crossing_deg(&c).unwrap();
        assert_close(deg, 27.196569908994878, 1e-12, "crossing");
        assert!((deg - PHI_HS_DEG_LCCO).abs() < 0.5);
        // A Fermi surface that misses the zone boundary has no CVC centres.
        let small = SfConfig { kf0: 4.0e9, ..c };
        assert!(zone_boundary_crossing_deg(&small).is_err());
        assert!(cvc_centres(&small).is_err());
        assert!(hall_coefficient(&small, 40.0, true).is_err());
        assert!(hall_coefficient(&small, 40.0, false).is_ok());
    }

    #[test]
    fn test_cvc_centres_follow_crossing() {
        let c = SfConfig::default();
        let centres = cvc_centres(&c).unwrap();
        let p = zone_boundary_crossing_deg(&c).unwrap().to_radians();
        let expected = [
            p,
            0.5 * PI - p,
            p + 0.5 * PI,
            PI - p,
            p + PI,
            1.5 * PI - p,
            p + 1.5 * PI,
            2.0 * PI - p,
        ];
        for (&got, want) in centres.iter().zip(expected) {
            assert_close(got, want, 1e-14, "centre");
        }
        // phi_HS (27.5 deg) is not the crossing, so the Gamma_HS centres differ.
        for (a, b) in centres.iter().zip(&hotspot_angles(&c)) {
            assert!((a - b).abs() > 1e-3);
        }
        for &phi in &centres {
            assert_close(
                cvc_epsilon_profile(phi, &centres, &c),
                c.cvc_epsilon,
                1e-12,
                "eps",
            );
        }
        // phi_HS moves only Gamma_HS and W: eps(phi) is unchanged.
        let moved = SfConfig {
            phi_hs_deg: 30.0,
            ..c
        };
        let moved_centres = cvc_centres(&moved).unwrap();
        for k in 0..64 {
            let phi = 0.1 * k as f64;
            assert_eq!(
                cvc_epsilon_profile(phi, &moved_centres, &moved),
                cvc_epsilon_profile(phi, &centres, &c)
            );
        }
        // J*_x at the first four centres in Fig. 12a's inward convention.
        let want = [-3.4356, 3.4356, -3.4356, 3.4356];
        let (jx, _) = cvc_velocity(&c).unwrap();
        assert_eq!(jx.len(), c.n_phi);
        for (&phi, w) in centres.iter().zip(want) {
            let eps = cvc_epsilon_profile(phi, &centres, &c);
            let vx = fermi_surface_at(phi, &c).3;
            let pvx = fermi_surface_at(cvc_partner_angle(phi, &c), &c).3;
            let inward = -(vx + eps * pvx) / 1e5;
            assert!((inward - w).abs() <= 0.15, "J*_x = {inward}");
        }
    }

    #[test]
    fn test_cvc_partner_maps_hotspots() {
        // The partner of each CVC centre is another centre.
        let c = SfConfig::default();
        let centres = cvc_centres(&c).unwrap();
        for &phi in &centres {
            let partner = cvc_partner_angle(phi, &c);
            let nearest = centres
                .iter()
                .map(|&q| wrap(partner - q).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest <= 1e-12, "phi = {phi}: {nearest:e}");
        }
    }

    #[test]
    fn test_cvc_partner_angle_zero_is_pi() {
        let c = SfConfig::default();
        assert_eq!(cvc_partner_angle(0.0, &c), PI);
        assert_eq!(cvc_partner_angle(-0.0, &c), PI);
    }

    // --- Scattering rate -----------------------------------------------------

    #[test]
    fn test_scattering_rate_peaks_at_hotspots() {
        let c = SfConfig::default();
        let gamma = |cfg: &SfConfig, phi: f64, t: f64| scattering_rate(phi, t, cfg).unwrap();
        for &phi in &hotspot_angles(&c) {
            let peak = gamma(&c, phi, 40.0);
            assert!(peak > gamma(&c, phi + 0.02, 40.0) && peak > gamma(&c, phi - 0.02, 40.0));
            // Gamma_HS adds the effective height on top of the hot-spot-free rate.
            let off = SfConfig {
                hotspots: false,
                ..c
            };
            assert_close(
                peak - gamma(&off, phi, 40.0),
                HOTSPOT_PEAK_LCCO,
                1e-6,
                "Gamma_HS",
            );
        }
        // Gamma(T) - Gamma(0) = (g alpha1 T + alpha2 T^2) W; g scales only the T-linear part.
        let c116 = SfConfig { doping: 0.116, ..c };
        for k in 0..16 {
            let phi = 0.1 + 0.37 * k as f64;
            let w = inelastic_weight(phi, &c);
            for (cfg, g) in [(c, 1.0), (c116, 6.50)] {
                let rise = gamma(&cfg, phi, 40.0) - gamma(&cfg, phi, 0.0);
                let want = (g * 0.75 * 40.0 + 0.0112 * 1600.0) * w;
                assert!(
                    (rise - want).abs() <= 1e-12 * gamma(&cfg, phi, 40.0),
                    "phi = {phi}"
                );
            }
        }
        assert!(scattering_rate(0.3, -1.0, &c).is_err());
    }

    #[test]
    fn test_form_phi_averages() {
        let c = SfConfig::default();
        let fs = fermi_surface(&c).unwrap();
        // Plain phi-average and DOS-weighted average per form (D3).
        let cases = [
            (
                SfInelasticForm::AfmBoundary,
                231.0 / 512.0,
                0.4580029405789852,
            ),
            (SfInelasticForm::Antinodal, 0.5, 0.4908459753825611),
            (SfInelasticForm::Isotropic, 1.0, 1.0),
        ];
        for (inelastic, plain, dos) in cases {
            let cf = SfConfig { inelastic, ..c };
            let w: Vec<f64> = fs.phi.iter().map(|&p| inelastic_weight(p, &cf)).collect();
            let mean = w.iter().sum::<f64>() / w.len() as f64;
            let weighted = w.iter().zip(&fs.weight).map(|(a, b)| a * b).sum::<f64>()
                / fs.weight.iter().sum::<f64>();
            assert_close(mean, plain, 1e-12, inelastic.as_str());
            assert_close(weighted, dos, 1e-12, inelastic.as_str());
        }
        // 2 C(nu, nu/2) / 2^nu for nu = 12.
        assert_eq!(2.0 * 924.0 / 4096.0, 231.0 / 512.0);
        // Antinodal is exactly cos^2(2 phi) at any nu; none is zero.
        for nu in [2, 8, 24] {
            let cn = SfConfig {
                inelastic: SfInelasticForm::Antinodal,
                nu,
                ..c
            };
            assert_eq!(inelastic_weight(0.3, &cn), (0.6_f64).cos().powi(2));
        }
        let none = SfConfig {
            inelastic: SfInelasticForm::None,
            ..c
        };
        assert_eq!(inelastic_weight(0.3, &none), 0.0);
        // Isotropic Gamma without hot spots is flat at Gamma_imp + g alpha1 T + alpha2 T^2.
        let iso = SfConfig {
            inelastic: SfInelasticForm::Isotropic,
            hotspots: false,
            ..c
        };
        let flat = GAMMA_IMP_LCCO + (0.75 * 40.0 + 0.0112 * (40.0 * 40.0));
        for k in 0..32 {
            assert_eq!(scattering_rate(0.2 * k as f64, 40.0, &iso).unwrap(), flat);
        }
    }

    // --- Chambers ------------------------------------------------------------

    #[test]
    fn test_circular_fs_drude_limits() {
        // n_phi = 1024: on this grid the integrator's O(h^4) error is 2e-12;
        // at 256 it is 5e-10, above the 1e-10 threshold.
        let circ = SfConfig {
            kf_anisotropy: 0.0,
            vf_anisotropy: 0.0,
            inelastic: SfInelasticForm::None,
            hotspots: false,
            ..Default::default()
        };
        let fs = fermi_surface(&circ).unwrap();
        let fields = [0.0, 2.0, 20.0, 200.0];
        let s = conductivity_tensor(&circ, 0.0, &fields).unwrap();
        let s0 = s[0][0][0];
        let tau = 1.0 / (circ.gamma_imp * 1e12);
        for (&b, sk) in fields.iter().zip(&s).skip(1) {
            let wt = fs.omega_per_tesla[0] * b * tau;
            assert_close(sk[0][0], s0 / (1.0 + wt * wt), 1e-10, "sigma_xx");
            assert_close(sk[1][0].abs(), s0 * wt / (1.0 + wt * wt), 1e-10, "sigma_yx");
            assert!(sk[0][1] > 0.0 && sk[1][0] < 0.0);
        }
        let r = resistivity_tensor(&circ, 0.0, &fields).unwrap();
        for rk in &r[1..] {
            assert!(
                (rk[0] / r[0][0] - 1.0).abs() <= 1e-10,
                "circle MR {}",
                rk[0] / r[0][0] - 1.0
            );
        }
    }

    #[test]
    fn test_onsager_symmetry() {
        let s = conductivity_tensor(&cfg256(), 40.0, &[20.0, -20.0]).unwrap();
        assert!((s[0][0][0] / s[1][0][0] - 1.0).abs() <= 1e-13);
        assert!((s[0][0][1] / (-s[1][0][1]) - 1.0).abs() <= 1e-13);
    }

    #[test]
    fn test_zero_field_closed_form() {
        let c = cfg256();
        for t in [2.0, 40.0, 100.0] {
            let closed = zero_field_resistivity(&c, &[t]).unwrap()[0];
            let tensor = resistivity_tensor(&c, t, &[0.0]).unwrap()[0][0];
            assert_close(tensor, closed, 1e-14, "B = 0");
        }
    }

    #[test]
    fn test_low_field_chambers_matches_jones_zener() {
        // I3: n_phi = 1024 (at 256 the O(|B| h) artefact is 1.1e-4 at 40 K).
        let c = SfConfig::default();
        let chambers = hall_coefficient_chambers(&c, 40.0, 0.1).unwrap();
        assert_close(chambers, rh(&c, 40.0, false), 1e-4, "default, 40 K");
        let iso = case_a(&c);
        let chambers_iso = hall_coefficient_chambers(&iso, 0.0, 0.1).unwrap();
        assert_close(chambers_iso, rh(&iso, 0.0, false), 1e-6, "isotropic");
    }

    #[test]
    fn test_sigma_xx_quadratic_at_low_field() {
        // I3: n_phi = 1024. No spurious H-linear term: (sigma_xx(B)/sigma_xx(0) - 1)/B^2
        // agrees at 1e-4 T and 0.1 T (4.4e-4 measured; 1.4e-3 at 256).
        let s = conductivity_tensor(&SfConfig::default(), 40.0, &[0.0, 1e-4, 0.1]).unwrap();
        let coeff = |k: usize, b: f64| (s[k][0][0] / s[0][0][0] - 1.0) / (b * b);
        assert!(coeff(1, 1e-4) < 0.0);
        assert!((coeff(2, 0.1) / coeff(1, 1e-4) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_mr_converges_with_n_phi() {
        let mr1024 = mr_at(&SfConfig::default(), 40.0, 20.0);
        let fine = SfConfig {
            n_phi: 2048,
            ..Default::default()
        };
        assert!((mr_at(&fine, 40.0, 20.0) / mr1024 - 1.0).abs() < 5e-4);
    }

    // --- Paper behaviour (n_phi = 256) ----------------------------------------

    #[test]
    fn test_anti_kohler() {
        let fam =
            magnetoresistance(&cfg256(), &SF_MR_TEMPERATURES, SF_FIELD_MAX, SF_FIELD_STEP).unwrap();
        assert_eq!(fam.fields.len(), 71);
        assert_eq!(fam.mr.len(), 4);
        let column = |k: usize| -> Vec<f64> { fam.mr.iter().map(|row| row[k]).collect() };
        let (i20, i33, i35) = (40, 66, 70);
        assert_eq!(fam.fields[i20], 20.0);
        assert!(strictly_increasing(&column(i20)) && strictly_increasing(&column(i35)));
        // Fig. 4b3 within 6 % (+1 % at n_phi = 256) and Fig. 4b2 within 1 %.
        for (v, paper) in column(i20).iter().zip([0.0195, 0.0211, 0.0234, 0.0260]) {
            assert!((v / paper - 1.0).abs() <= 0.07, "MR(20 T) {v}");
        }
        for (v, paper) in column(i33).iter().zip([0.0387, 0.0444, 0.0505, 0.0564]) {
            assert!((v / paper - 1.0).abs() <= 0.07, "MR(33 T) {v}");
        }
        for (row, paper) in fam.rho_xx.iter().zip([19.99, 22.15, 24.42, 26.78]) {
            assert!((row[0] / paper - 1.0).abs() <= 0.01, "rho(0) {}", row[0]);
        }
        // Bookkeeping: MR(0) = 0, H/T and the derivative stencil.
        assert!(fam.mr.iter().all(|row| row[0] == 0.0));
        assert_eq!(fam.h_over_t[0][i20], 1.0);
        let d = &fam.drho_dh[2];
        let r = &fam.rho_xx[2];
        assert_eq!(d[10], (r[11] - r[9]) / (2.0 * SF_FIELD_STEP));
        // The family equals independent per-temperature evaluations.
        let direct = resistivity_tensor(&cfg256(), 30.0, &fam.fields).unwrap();
        for (a, b) in direct.iter().zip(fam.rho_xx[1].iter().zip(&fam.rho_yx[1])) {
            assert_eq!((a[0], a[1]), (*b.0, *b.1));
        }
    }

    #[test]
    fn test_isotropic_is_kohler_like() {
        let k = SfConfig {
            inelastic: SfInelasticForm::Isotropic,
            hotspots: false,
            ..cfg256()
        };
        let v: Vec<f64> = SF_MR_TEMPERATURES
            .iter()
            .map(|&t| mr_at(&k, t, 20.0))
            .collect();
        assert!(strictly_decreasing(&v), "{v:?}");
    }

    #[test]
    fn test_kohler_collapse_uniform_gamma() {
        // Uniform Gamma: MR depends on B and T only through B / rho0.
        let k = SfConfig {
            inelastic: SfInelasticForm::Isotropic,
            hotspots: false,
            ..cfg256()
        };
        let at = |t: f64| {
            let rho0 = zero_field_resistivity(&k, &[t]).unwrap()[0];
            mr_at(&k, t, rho0)
        };
        let (cold, hot) = (at(20.0), at(50.0));
        assert!((cold / hot - 1.0).abs() <= 1e-10, "{cold} vs {hot}");
    }

    #[test]
    fn test_fig9c_inverts_t_dependence() {
        let c9 = SfConfig {
            inelastic: SfInelasticForm::Isotropic,
            hotspot_peak: HOTSPOT_PEAK_FIG9C,
            ..cfg256()
        };
        let v: Vec<f64> = SF_MR_TEMPERATURES
            .iter()
            .map(|&t| mr_at(&c9, t, 35.0))
            .collect();
        assert!(strictly_decreasing(&v), "{v:?}");
        let rho = zero_field_resistivity(&c9, &[20.0, 50.0]).unwrap();
        assert!(rho[1] / rho[0] > 1.8);
    }

    #[test]
    fn test_fig9e_antinodal_saturates() {
        // slope(35 T) / max slope at 20 K < 0.75 and below the paper form's.
        let ratio = |cfg: &SfConfig| {
            let fam = magnetoresistance(cfg, &[20.0], SF_FIELD_MAX, SF_FIELD_STEP).unwrap();
            let d = &fam.drho_dh[0];
            d[d.len() - 1] / d.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        };
        let antinodal = SfConfig {
            inelastic: SfInelasticForm::Antinodal,
            ..cfg256()
        };
        let (an, paper) = (ratio(&antinodal), ratio(&cfg256()));
        assert!(an < 0.75 && an < paper, "{an} vs {paper}");
    }

    #[test]
    fn test_no_hotspots_fans_initial_slopes() {
        // Fig. 9b: weaker low-T MR and initial slopes that fan out with T.
        let no_hs = SfConfig {
            hotspots: false,
            ..cfg256()
        };
        assert!(mr_at(&no_hs, 20.0, 35.0) < 0.6 * mr_at(&cfg256(), 20.0, 35.0));
        let fan = |cfg: &SfConfig| {
            let fam = magnetoresistance(cfg, &[20.0, 50.0], SF_FIELD_MAX, SF_FIELD_STEP).unwrap();
            fam.drho_dh[1][5] / fam.drho_dh[0][5]
        };
        assert!(fan(&no_hs) > 2.5);
        assert!(fan(&cfg256()) < 2.0);
    }

    #[test]
    fn test_no_h_over_t_collapse() {
        // d rho / d mu0H at mu0H / T = 0.5 spreads > 2x over 20-50 K (Fig. 2h-l).
        let c = cfg256();
        let slopes: Vec<f64> = SF_MR_TEMPERATURES
            .iter()
            .map(|&t| high_field_slope(&c, t, 0.5 * t, 0.5).unwrap())
            .collect();
        assert!(spread(&slopes) > 2.0, "{slopes:?}");
    }

    #[test]
    fn test_mr_falls_with_doping() {
        let c = cfg256();
        let sweep = doping_sweep(&c, SF_FIT_TEMPERATURE, SF_FIT_FIELD).unwrap();
        assert_eq!(sweep.len(), 19);
        let s1: Vec<&SfDopingPoint> = sweep.iter().filter(|p| p.doping <= SF_DOPING_REF).collect();
        assert_eq!(s1.len(), 13);
        let mr: Vec<f64> = s1.iter().map(|p| p.mr).collect();
        let rho: Vec<f64> = s1.iter().map(|p| p.rho0).collect();
        assert!(strictly_decreasing(&mr) && strictly_decreasing(&rho));
        assert!(sweep.windows(2).all(|w| w[1].mr <= w[0].mr));
        for ratio in [mr[0] / mr[12], rho[0] / rho[12]] {
            assert!((1.5..=2.0).contains(&ratio), "S1 drop {ratio}");
        }
        // Fig. 5a/b: rho within 5 %, MR within 13 % (+1 % at n_phi = 256).
        let fig5 = [
            (0.116, 43.60, 0.0421),
            (0.120, 39.60, 0.0409),
            (0.125, 35.20, 0.0368),
            (0.131, 31.90, 0.0334),
            (0.138, 29.00, 0.0300),
            (0.146, 26.40, 0.0271),
            (0.153, 26.00, 0.0259),
            (0.159, 24.40, 0.0239),
        ];
        for (x, rho_paper, mr_paper) in fig5 {
            let p = sweep.iter().find(|p| p.doping == x).unwrap();
            assert!((p.rho0 / rho_paper - 1.0).abs() <= 0.05, "rho at {x}");
            assert!((p.mr / mr_paper - 1.0).abs() <= 0.14, "MR at {x}");
        }
        // Rows equal direct evaluations (same kernel arithmetic).
        let row = &sweep[12];
        assert_eq!(row.mr, mr_at(&c, SF_FIT_TEMPERATURE, SF_FIT_FIELD));
        assert_eq!(row.lambda_tr, lambda_tr(&c).unwrap());
        assert_eq!(row.lambda_rho, lambda_resistive(&c).unwrap());
        assert_eq!((row.g, row.tc), (1.0, tc_of_x(SF_DOPING_REF)));
    }

    #[test]
    fn test_table2_height_breaks_anti_kohler() {
        let c50 = SfConfig {
            hotspot_peak: HOTSPOT_PEAK_TABLE2,
            ..cfg256()
        };
        assert!(mr_at(&c50, 20.0, 20.0) > mr_at(&c50, 30.0, 20.0));
    }

    // --- Hall ----------------------------------------------------------------

    #[test]
    fn test_hall_fingerprints() {
        let c = cfg256();
        let checks = [2.0, 20.0, 50.0, 100.0, 200.0];
        for inelastic in SfInelasticForm::ALL {
            for hotspots in [true, false] {
                let cf = SfConfig {
                    inelastic,
                    hotspots,
                    ..c
                };
                let v = hall_coefficient_curve(&cf, &checks, false).unwrap();
                assert!(
                    v.iter().all(|&r| r > 0.0),
                    "{} {hotspots}",
                    inelastic.as_str()
                );
            }
        }
        let t2_200: Vec<f64> = (2..=200).map(f64::from).collect();
        assert!(strictly_decreasing(
            &hall_coefficient_curve(&c, &t2_200, false).unwrap()
        ));
        let antinodal = SfConfig {
            inelastic: SfInelasticForm::Antinodal,
            ..c
        };
        for cvc in [false, true] {
            let v = hall_coefficient_curve(&antinodal, &t2_200, cvc).unwrap();
            assert!(strictly_increasing(&v), "antinodal, cvc = {cvc}");
        }
        // CVCs lower (a) 7.340 -> 5.309, (b) 7.278 -> 4.929 and (d) at 2-100 K.
        let a = case_a(&c);
        let b = SfConfig {
            inelastic: SfInelasticForm::None,
            ..c
        };
        for (cfg, plain, dressed) in [(a, 7.340, 5.309), (b, 7.278, 4.929)] {
            assert!((rh(&cfg, 0.0, false) * 1e10 - plain).abs() < 5e-4);
            assert!((rh(&cfg, 0.0, true) * 1e10 - dressed).abs() < 5e-4);
        }
        let t2_100: Vec<f64> = (2..=100).map(f64::from).collect();
        let d_cvc = hall_coefficient_curve(&c, &t2_100, true).unwrap();
        let d_plain = hall_coefficient_curve(&c, &t2_100, false).unwrap();
        assert!(d_cvc.iter().zip(&d_plain).all(|(x, y)| x < y));
        for t in [2.0, 50.0, 100.0] {
            let ratio = rh(&c, t, true) / rh(&a, t, true);
            assert!((0.85..=1.0).contains(&ratio), "(d)/(a) at {t} K = {ratio}");
        }
        // Isotropic without CVC within 2 % of Drude.
        let drude = drude_hall_coefficient(&c).unwrap();
        assert!((rh(&a, 0.0, false) / drude - 1.0).abs() < 0.02);
        assert_close(
            drude_hall_coefficient_nominal(&SfConfig::default()).unwrap(),
            7.3871e-10,
            1e-4,
            "nominal Drude at 0.159",
        );
        let x116 = SfConfig {
            doping: 0.116,
            ..Default::default()
        };
        assert_close(
            drude_hall_coefficient_nominal(&x116).unwrap(),
            7.0278e-10,
            1e-4,
            "nominal Drude at 0.116",
        );
    }

    #[test]
    fn test_hall_cvc_flattens_t_dependence() {
        let c = cfg256();
        let (r30, r40, r100) = (rh(&c, 30.0, true), rh(&c, 40.0, true), rh(&c, 100.0, true));
        assert!(r100 < r40 && r40 < r30);
        let temps: Vec<f64> = (2..=100).map(f64::from).collect();
        let with = spread(&hall_coefficient_curve(&c, &temps, true).unwrap());
        let without = spread(&hall_coefficient_curve(&c, &temps, false).unwrap());
        assert!(with < 1.10 && without > 1.40, "{with} / {without}");
    }

    #[test]
    fn test_hall_fig13_order_at_100k() {
        let d = cfg256();
        let a = case_a(&d);
        let b = SfConfig {
            inelastic: SfInelasticForm::None,
            ..d
        };
        let c = SfConfig {
            inelastic: SfInelasticForm::Antinodal,
            ..d
        };
        let ordered = |t: f64, cvc: bool| {
            let v = [
                rh(&c, t, cvc),
                rh(&a, t, cvc),
                rh(&b, t, cvc),
                rh(&d, t, cvc),
            ];
            strictly_decreasing(&v)
        };
        assert!(ordered(100.0, true));
        assert!(ordered(50.0, false) && ordered(100.0, false));
        // With CVC, (d) > (b) below 66.9 K (flag F3).
        assert!(rh(&d, 50.0, true) > rh(&b, 50.0, true));
    }

    #[test]
    fn test_hall_ratio_to_case_a() {
        let c = SfConfig::default();
        let a = case_a(&c);
        for t in [0.0, 40.0, 200.0] {
            assert_close(
                rh(&a, t, true),
                5.309220240734655e-10,
                1e-12,
                "case (a) with CVC",
            );
        }
        assert_close(
            hall_ratio_to_case_a(&c, 40.0).unwrap(),
            0.9592004336484117,
            1e-12,
            "P6e",
        );
        // Not the isotropic form with hot spots, whose R_H is T-dependent.
        let iso_hs = SfConfig {
            inelastic: SfInelasticForm::Isotropic,
            ..c
        };
        assert!((rh(&iso_hs, 40.0, true) / rh(&a, 40.0, true) - 1.0).abs() > 1e-3);
    }

    // --- Coupling ------------------------------------------------------------

    #[test]
    fn test_lambda_tr_coefficient() {
        let c = SfConfig::default();
        assert_close(
            lambda_tr(&c).unwrap(),
            0.417582739923252,
            1e-12,
            "coefficient",
        );
        // An independent DOS-weighted quadrature at N = 8192.
        let n = 8192;
        let (mut num, mut den) = (0.0, 0.0);
        for m in 0..n {
            let phi = 2.0 * PI * m as f64 / n as f64;
            let (kf, vf, gamma, _, _) = fermi_surface_at(phi, &c);
            let w = kf / (vf * gamma.cos());
            num += inelastic_weight(phi, &c) * w;
            den += w;
        }
        let independent = HBAR_J_S * 0.75e12 * (num / den) / (2.0 * PI * KB_J_K);
        assert_close(lambda_tr(&c).unwrap(), independent, 1e-12, "N = 8192");
        // Per form (D3) and strictly increasing in g.
        let per_form = [
            (SfInelasticForm::AfmBoundary, 0.41758273992325223),
            (SfInelasticForm::Antinodal, 0.4475272735616924),
            (SfInelasticForm::Isotropic, 0.9117468534052734),
        ];
        for (inelastic, want) in per_form {
            assert_close(
                lambda_tr(&SfConfig { inelastic, ..c }).unwrap(),
                want,
                1e-12,
                inelastic.as_str(),
            );
        }
        let none = SfConfig {
            inelastic: SfInelasticForm::None,
            ..c
        };
        assert_eq!(lambda_tr(&none).unwrap(), 0.0);
        let at = |doping: f64| lambda_tr(&SfConfig { doping, ..c }).unwrap();
        assert!(at(0.116) > at(0.141) && at(0.141) > at(0.159) && at(0.159) > at(0.188));
    }

    #[test]
    fn test_lambda_resistive_matches_isotropic() {
        // Uniform Gamma: rho = rho_D Gamma exactly, so lambda_rho = lambda_tr(isotropic).
        for alpha2 in [0.0, ALPHA2_LCCO] {
            let iso = SfConfig {
                inelastic: SfInelasticForm::Isotropic,
                hotspots: false,
                alpha2,
                ..Default::default()
            };
            assert_close(
                lambda_resistive(&iso).unwrap(),
                lambda_tr(&iso).unwrap(),
                1e-9,
                "uniform",
            );
        }
    }

    #[test]
    fn test_lambda_rho_table() {
        let c = SfConfig::default();
        assert_close(
            drude_resistivity_per_rate(&c).unwrap(),
            5.5005672382590704e-21,
            1e-12,
            "rho_D",
        );
        let sweep = doping_sweep(&c, SF_FIT_TEMPERATURE, SF_FIT_FIELD).unwrap();
        let ratio: Vec<f64> = sweep.iter().map(|p| p.lambda_rho / p.lambda_tr).collect();
        assert!(ratio.windows(2).all(|w| w[1] >= w[0]), "{ratio:?}");
        let (first, last) = (sweep[0], sweep[18]);
        assert!((first.lambda_tr / last.lambda_tr - 11.82).abs() <= 0.01);
        assert!((first.lambda_rho / last.lambda_rho - 5.06).abs() <= 0.01);
    }

    #[test]
    fn test_equivalent_doping() {
        let lambda_bar = lambda_from_tc(13.5, &SfPairingMap::default()).unwrap();
        assert_close(
            equivalent_doping(lambda_bar).unwrap(),
            0.1435584333754864,
            1e-12,
            "x_eq",
        );
        assert_close(
            equivalent_doping(SF_POWER_LAW_LAMBDA_C).unwrap(),
            0.17052577242235883,
            1e-12,
            "x_c",
        );
        for outside in [3.0, 0.2, f64::NAN] {
            assert!(equivalent_doping(outside).is_none(), "lambda = {outside}");
        }
        let at_ref = equivalent_doping(lambda_tr(&SfConfig::default()).unwrap()).unwrap();
        assert_close(at_ref, SF_DOPING_REF, 1e-12, "x at g = 1");
    }

    // --- Pairing map -----------------------------------------------------------

    #[test]
    fn test_power_law_hits_labels() {
        let pl = SfPairingMap::default();
        let points = calibration_points();
        assert_close(points[0].2, SF_POWER_LAW_LAMBDA_REF, 1e-12, "lambda(0.12)");
        for &(x, tc, lam) in &points[..3] {
            assert_close(tc_from_lambda(lam, &pl), tc, 1e-9, &format!("label {x}"));
        }
        assert!(points[3].2 < SF_POWER_LAW_LAMBDA_C);
        assert_eq!(tc_from_lambda(points[3].2, &pl), 0.0);
        assert!(pairing_fit_rms(&pl) < 1e-6);
    }

    #[test]
    fn test_mcmillan_fit_is_local_minimum() {
        let mc = mcmillan();
        let rms = pairing_fit_rms(&mc);
        assert!((rms - 2.907).abs() <= 1e-3, "rms = {rms}");
        for omega_k in [mc.omega_k - 0.5, mc.omega_k + 0.5] {
            assert!(pairing_fit_rms(&SfPairingMap { omega_k, ..mc }) > rms);
        }
        assert!(
            pairing_fit_rms(&SfPairingMap {
                mu_star: 0.01,
                ..mc
            }) > rms
        );
    }

    #[test]
    fn test_inverse_roundtrip() {
        for p in [SfPairingMap::default(), mcmillan()] {
            for lam in [0.5, 0.9, 1.6, 2.7] {
                let tc = tc_from_lambda(lam, &p);
                assert_close(lambda_from_tc(tc, &p).unwrap(), lam, 1e-12, p.form.as_str());
            }
            let threshold = lambda_threshold(&p);
            assert_eq!(tc_from_lambda(threshold, &p), 0.0);
            assert_eq!(tc_from_lambda(threshold - 0.01, &p), 0.0);
            assert!(lambda_from_tc(0.0, &p).is_err() && lambda_from_tc(-1.0, &p).is_err());
        }
        let mc = mcmillan();
        assert_close(tc_ceiling(&mc), 43.39985, 1e-6, "McMillan ceiling");
        assert!(lambda_from_tc(tc_ceiling(&mc), &mc).is_err());
        assert!(lambda_from_tc(50.0, &mc).is_err());
        assert!(tc_ceiling(&SfPairingMap::default()).is_infinite());
    }

    #[test]
    fn test_unresolvable_tc_target_is_rejected() {
        // Python's test_unresolvable_tc_raises: the power law's inverse rounds
        // onto lambda_c at and below 6.173098052100439e-06 K, a flat exponent
        // does so at 5 K, and McMillan's 1.45 Tc / omega underflows at 1e-323 K
        // (lambda = NaN; Python's math.log raises there).
        let pl = SfPairingMap::default();
        let mc = mcmillan();
        let flat = SfPairingMap {
            exponent: 0.001,
            ..Default::default()
        };
        let rejected = [
            (pl, 6.173098052100439e-06),
            (pl, 1e-6),
            (pl, 1e-8),
            (flat, 5.0),
            (mc, 1e-323),
        ];
        for (p, tc) in rejected {
            let err = lambda_from_tc(tc, &p).unwrap_err();
            assert!(err.contains("resolvable range"), "{tc} K: {err}");
            assert!(local_imprint(0.1, tc, &p).is_err(), "{tc} K");
            // Every glue entry point validates through the same check.
            for coherence in SfCoherence::ALL {
                let glue = SfGlueConfig {
                    tc_target: tc,
                    coherence,
                    pairing: p,
                    ..Default::default()
                };
                let what = format!("{tc} K, {}", coherence.as_str());
                assert!(
                    glue_modulated_gap(&[0.0, 0.5, 1.0], &glue).is_err(),
                    "{what}"
                );
                assert!(coherence_length(&glue).is_err(), "{what}");
                assert!(coherence_factor(&glue).is_err(), "{what}");
                assert!(glue_gap_scale(&glue).is_err(), "{what}");
                assert!(linearized_glue_amplitude(&glue).is_err(), "{what}");
                assert!(imprint_curve(&glue, &[tc]).is_err(), "{what}");
                assert!(fragmentation_tc(&glue).is_err(), "{what}");
            }
        }
        // A bad target anywhere in the list fails the whole curve.
        assert!(imprint_curve(&SfGlueConfig::default(), &[13.5, 1e-6]).is_err());
        // One ulp above the boundary the inverse clears lambda_c (both stacks:
        // lambda = 0.304620157769113, Tc(lambda) = 8.104066120908808e-06 K).
        let tc = 6.17309805210044e-06;
        let lam = lambda_from_tc(tc, &pl).unwrap();
        assert!(lam > pl.lambda_c && tc_from_lambda(lam, &pl) > 0.0);
        assert!(local_imprint(0.1, tc, &pl).is_ok());
        for coherence in SfCoherence::ALL {
            let glue = SfGlueConfig {
                tc_target: tc,
                coherence,
                ..Default::default()
            };
            assert!(glue_modulated_gap(&[0.0, 0.5, 1.0], &glue).is_ok());
            assert!(imprint_curve(&glue, &[tc]).is_ok());
        }
        // McMillan still resolves at 1e-300 K; the range test keeps its own
        // message and runs first.
        assert!(tc_from_lambda(lambda_from_tc(1e-300, &mc).unwrap(), &mc) > 0.0);
        for tc in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(lambda_from_tc(tc, &pl).unwrap_err().contains("must lie in"));
        }
        assert!(lambda_from_tc(50.0, &mc)
            .unwrap_err()
            .contains("must lie in"));
    }

    #[test]
    fn test_log_sensitivity_finite_difference() {
        for p in [SfPairingMap::default(), mcmillan()] {
            for lam in [0.4, 0.7, 1.5, 2.5] {
                let h = 1e-6 * lam;
                let fd = (tc_from_lambda(lam + h, &p).ln() - tc_from_lambda(lam - h, &p).ln())
                    / ((lam + h).ln() - (lam - h).ln());
                assert_close(log_sensitivity(lam, &p), fd, 1e-6, p.form.as_str());
            }
        }
    }

    // --- Glue what-if ----------------------------------------------------------

    #[test]
    fn test_glue_uniform_pattern_gives_target() {
        let half = vec![0.5; 64];
        for coherence in SfCoherence::ALL {
            for tc_target in [2.0, 5.0, 13.5] {
                let glue = SfGlueConfig {
                    tc_target,
                    coherence,
                    ..Default::default()
                };
                let res = glue_modulated_gap(&half, &glue).unwrap();
                let delta_bar = glue_gap_scale(&glue).unwrap();
                assert!(res
                    .gap
                    .iter()
                    .all(|&g| (g - delta_bar).abs() <= 1e-12 * delta_bar));
            }
        }
        let flat = SfGlueConfig {
            eta: 0.0,
            ..Default::default()
        };
        let res = glue_modulated_gap(&ramp_pattern(101), &flat).unwrap();
        assert!(res
            .gap
            .iter()
            .all(|&g| (g - DELTA_AVG).abs() <= 1e-15 * DELTA_AVG));
        assert!(res.relative_modulation <= 1e-15);
    }

    #[test]
    fn test_glue_small_eta_matches_linear_route() {
        // eta = 1e-4: the glue map tends to gap_modulation(pattern, Delta_bar,
        // linearized_glue_amplitude) = Delta_bar + A cos(pi + pi m).
        let pattern = ramp_pattern(257);
        for coherence in SfCoherence::ALL {
            for tc_target in [13.5, 5.0] {
                let glue = SfGlueConfig {
                    eta: 1e-4,
                    tc_target,
                    coherence,
                    ..Default::default()
                };
                let res = glue_modulated_gap(&pattern, &glue).unwrap();
                let delta_bar = glue_gap_scale(&glue).unwrap();
                let amplitude = linearized_glue_amplitude(&glue).unwrap();
                for (&m, &gap) in pattern.iter().zip(&res.gap) {
                    let linear = delta_bar + amplitude * (PI + PI * m).cos();
                    assert_close(gap, linear, 1e-6, coherence.as_str());
                }
            }
        }
    }

    #[test]
    fn test_glue_local_limit_fragments() {
        let pattern = ramp_pattern(201);
        let frag = fragmentation_tc(&SfGlueConfig::default()).unwrap();
        assert_close(frag, 5.177318594015492, 1e-12, "Tc_frag");
        let run = |tc_target: f64, coherence: SfCoherence| {
            let glue = SfGlueConfig {
                tc_target,
                coherence,
                ..Default::default()
            };
            glue_modulated_gap(&pattern, &glue).unwrap()
        };
        assert!(run(5.1, SfCoherence::Local).local_normal_fraction > 0.0);
        assert_eq!(run(5.2, SfCoherence::Local).local_normal_fraction, 0.0);
        // Tc(r) = 0 somewhere iff lambda_bar (1 - eta) <= lambda_c.
        for tc_target in [2.0, 5.0, 5.17, 5.18, 8.0, 13.5] {
            let res = run(tc_target, SfCoherence::Local);
            let fragments = res.lambda_bar * (1.0 - SF_GLUE_ETA) <= SF_POWER_LAW_LAMBDA_C;
            assert_eq!(
                res.local_normal_fraction > 0.0,
                fragments,
                "Tc* = {tc_target}"
            );
            assert_eq!(res.normal_fraction, res.local_normal_fraction);
            for coherence in [SfCoherence::Clean, SfCoherence::Dirty] {
                assert_eq!(run(tc_target, coherence).normal_fraction, 0.0);
            }
        }
        // Dressed gaps stay positive; the local map closes the gap in places.
        assert!(run(2.0, SfCoherence::Clean).gap.iter().all(|&g| g > 0.0));
        assert!(run(2.0, SfCoherence::Local).gap.contains(&0.0));
        let mc_glue = SfGlueConfig {
            pairing: mcmillan(),
            ..Default::default()
        };
        assert_eq!(fragmentation_tc(&mc_glue).unwrap(), 0.0);
    }

    #[test]
    fn test_glue_coherence_models() {
        let cases = [
            (13.5, 20.0, 20.0),
            (5.0, 54.0, 32.86335345030997),
            (2.0, 135.0, 51.96152422706632),
        ];
        for (tc_target, clean, dirty) in cases {
            let glue = |coherence| SfGlueConfig {
                tc_target,
                coherence,
                ..Default::default()
            };
            assert_close(
                coherence_length(&glue(SfCoherence::Clean)).unwrap(),
                clean,
                1e-14,
                "clean xi",
            );
            assert_close(
                coherence_length(&glue(SfCoherence::Dirty)).unwrap(),
                dirty,
                1e-14,
                "dirty xi",
            );
            assert_eq!(coherence_length(&glue(SfCoherence::Local)).unwrap(), clean);
            assert_eq!(coherence_factor(&glue(SfCoherence::Local)).unwrap(), 1.0);
            let res = glue_modulated_gap(&ramp_pattern(64), &glue(SfCoherence::Local)).unwrap();
            assert_eq!((res.coherence_length, res.coherence_factor), (clean, 1.0));
        }
        // xi / L_m = 0.545 > 1/2 at the defaults: the card flags the local map.
        let glue = SfGlueConfig::default();
        let xi_over_lm = coherence_length(&glue).unwrap() / glue.moire_period;
        assert!((xi_over_lm - 0.545).abs() < 1e-3);
    }

    #[test]
    fn test_glue_gap_scale_keeps_ratio() {
        // 2 Delta_bar / (k_B Tc*) with config.KB_EV_K = 8.617333262e-5 (pdw.rs's literal).
        for tc_target in [2.0, 5.0, 13.5] {
            let glue = SfGlueConfig {
                tc_target,
                ..Default::default()
            };
            let delta_bar = glue_gap_scale(&glue).unwrap();
            let ratio = 2.0 * delta_bar * 1e-3 / (8.617333262e-5 * tc_target);
            assert_close(ratio, 5.312290517954645, 1e-12, "2 Delta / kB Tc");
        }
        assert_eq!(glue_gap_scale(&SfGlueConfig::default()).unwrap(), DELTA_AVG);
    }

    #[test]
    fn test_glue_dressed_identity_and_coherence_factor() {
        let lm = SF_GLUE_MOIRE_PERIOD;
        for tc_target in [2.0, 5.0, 13.5, 0.05] {
            let glue = SfGlueConfig {
                tc_target,
                ..Default::default()
            };
            let xi = coherence_length(&glue).unwrap();
            assert_eq!(coherence_factor(&glue).unwrap(), (-(xi / lm)).exp());
        }
        // xi / L_m = 147 at 0.05 K: no clamp (a clamped exponent would give 3.72e-44).
        let cold = SfGlueConfig {
            tc_target: 0.05,
            ..Default::default()
        };
        assert_close(
            coherence_factor(&cold).unwrap(),
            1.1845830137647655e-64,
            1e-12,
            "F(0.05 K)",
        );
        // relative = F x local relative = the pattern-free dressed imprint.
        let pattern = ramp_pattern(401);
        for coherence in SfCoherence::ALL {
            for tc_target in [2.0, 5.0, 13.5] {
                let glue = SfGlueConfig {
                    tc_target,
                    coherence,
                    ..Default::default()
                };
                let res = glue_modulated_gap(&pattern, &glue).unwrap();
                let product = res.coherence_factor * res.local_relative_modulation;
                assert_close(res.relative_modulation, product, 1e-12, "F x local");
                let (dressed, local, _) = imprint_curve(&glue, &[tc_target]).unwrap();
                assert_close(res.relative_modulation, dressed[0], 1e-12, "pattern-free");
                assert_eq!(dressed[0], res.coherence_factor * local[0]);
            }
        }
    }

    #[test]
    fn test_glue_default_period_is_shared() {
        let lm = moire_locked_period(
            by_name("FeTe").expect("registry").a,
            by_name("Sb2Te3").expect("registry").a,
            0.0,
        )
        .expect("finite");
        assert_eq!(SF_GLUE_MOIRE_PERIOD, lm);
        assert_eq!(SfGlueConfig::default().moire_period, lm);
    }

    #[test]
    fn test_glue_clean_exceeds_reference_at_longer_period() {
        let targets = linspace(0.5, 25.0, 80);
        let reference = bcs_route_relative_modulation();
        let (at_shared, _, _) = imprint_curve(&SfGlueConfig::default(), &targets).unwrap();
        assert!(at_shared.iter().all(|&v| v < reference));
        // L_m condition (K2-1): at twice the shared period the clean curve crosses 0.165.
        let longer = SfGlueConfig {
            moire_period: 2.0 * SF_GLUE_MOIRE_PERIOD,
            ..Default::default()
        };
        let (dressed, _, _) = imprint_curve(&longer, &targets).unwrap();
        assert_eq!(dressed.iter().filter(|&&v| v > reference).count(), 15);
        let max = dressed.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_close(max, 0.32205266828326645, 1e-9, "max at 2 L_m");
    }

    #[test]
    fn test_imprint_curve_clean_peaks_at_fragmentation() {
        let targets = linspace(4.0, 7.0, 3001);
        let (dressed, _, _) = imprint_curve(&SfGlueConfig::default(), &targets).unwrap();
        let argmax = dressed
            .iter()
            .enumerate()
            .fold(0, |best, (k, &v)| if v > dressed[best] { k } else { best });
        let frag = fragmentation_tc(&SfGlueConfig::default()).unwrap();
        assert!(
            (targets[argmax] - frag).abs() <= 2e-3,
            "{} vs {frag}",
            targets[argmax]
        );
        let reference = bcs_route_relative_modulation();
        let one = |coherence, tc_target| {
            let glue = SfGlueConfig {
                coherence,
                ..Default::default()
            };
            imprint_curve(&glue, &[tc_target]).unwrap().0[0]
        };
        assert!(one(SfCoherence::Dirty, 2.0) > reference);
        assert!(one(SfCoherence::Local, 5.0) > reference);
        let mc_glue = SfGlueConfig {
            pairing: mcmillan(),
            ..Default::default()
        };
        assert_eq!(fragmentation_tc(&mc_glue).unwrap(), 0.0);
        let (mc_curve, _, _) = imprint_curve(&mc_glue, &linspace(0.5, 25.0, 80)).unwrap();
        assert!(mc_curve.iter().all(|&v| v < reference));
        assert_eq!(reference, 0.51 / 3.09);
    }

    #[test]
    fn test_quadrature_collapses() {
        let q = SfQuadratureConfig::default();
        for x in [0.1, 0.5, 2.0] {
            let base = quadrature_field_derivative(20.0, 20.0 * x, &q);
            for t in [5.0, 40.0, 75.0] {
                assert_close(
                    quadrature_field_derivative(t, t * x, &q),
                    base,
                    1e-12,
                    "H/T collapse",
                );
            }
        }
        assert_close(
            quadrature_field_derivative(1.0, 1e6, &q),
            q.gamma_mub,
            1e-9,
            "saturation",
        );
        assert_eq!(quadrature_field_derivative(40.0, 0.0, &q), 0.0);
        // 0 where the root vanishes or underflows, as Python, not 0/0 = NaN or
        // x/0 = inf; T = 0 alone keeps the gamma muB line, NaN stays NaN.
        for (t, h) in [(0.0, 0.0), (1e-300, 0.0), (0.0, 1e-300)] {
            assert_eq!(
                quadrature_field_derivative(t, h, &q),
                0.0,
                "T = {t:e}, H = {h:e}"
            );
        }
        assert_close(
            quadrature_field_derivative(0.0, 5.0, &q),
            q.gamma_mub,
            1e-15,
            "T = 0",
        );
        assert!(quadrature_field_derivative(f64::NAN, 1.0, &q).is_nan());
    }

    #[test]
    fn test_invalid_configs() {
        let base = SfConfig::default();
        let cases = [
            SfConfig {
                doping: 0.1,
                ..base
            },
            SfConfig {
                doping: 0.2,
                ..base
            },
            SfConfig {
                doping: f64::NAN,
                ..base
            },
            SfConfig {
                gamma_imp: 0.0,
                ..base
            },
            SfConfig {
                gamma_imp: f64::NAN,
                ..base
            },
            SfConfig {
                alpha1: -0.1,
                ..base
            },
            SfConfig {
                alpha2: f64::NAN,
                ..base
            },
            SfConfig {
                hotspot_peak: -1.0,
                ..base
            },
            SfConfig {
                hotspot_width: 0.0,
                ..base
            },
            SfConfig {
                cvc_sigma: f64::NAN,
                ..base
            },
            SfConfig { kf0: 0.0, ..base },
            SfConfig {
                vf0: f64::INFINITY,
                ..base
            },
            SfConfig {
                layer_spacing: -6.19,
                ..base
            },
            SfConfig { nu: 0, ..base },
            SfConfig { nu: 13, ..base },
            SfConfig { nu: 26, ..base },
            SfConfig {
                phi_hs_deg: 0.0,
                ..base
            },
            SfConfig {
                phi_hs_deg: 45.0,
                ..base
            },
            SfConfig {
                phi_hs_deg: f64::NAN,
                ..base
            },
            SfConfig {
                kf_anisotropy: 1.0,
                ..base
            },
            SfConfig {
                vf_anisotropy: -0.01,
                ..base
            },
            SfConfig {
                cvc_epsilon: 1.5,
                ..base
            },
            SfConfig {
                cvc_epsilon: f64::NAN,
                ..base
            },
            SfConfig { n_phi: 100, ..base },
            SfConfig { n_phi: 32, ..base },
            SfConfig {
                n_phi: 16384,
                ..base
            },
        ];
        for cfg in &cases {
            assert!(fermi_surface(cfg).is_err(), "cfg = {cfg:?}");
            assert!(lambda_tr(cfg).is_err(), "cfg = {cfg:?}");
            assert!(
                resistivity_tensor(cfg, 40.0, &[0.0, 20.0]).is_err(),
                "cfg = {cfg:?}"
            );
            assert!(hall_coefficient(cfg, 40.0, true).is_err(), "cfg = {cfg:?}");
        }
        // The doping sweep validates the whole config (as the Python twin does),
        // and its rows do not depend on the config's own doping.
        assert!(doping_sweep(&cases[2], 40.0, 20.0).is_err());
        assert!(doping_sweep(&cases[3], 40.0, 20.0).is_err());
        let at_0116 = SfConfig {
            doping: 0.116,
            n_phi: 64,
            ..base
        };
        let at_ref = SfConfig { n_phi: 64, ..base };
        assert_eq!(
            doping_sweep(&at_0116, 40.0, 20.0).unwrap(),
            doping_sweep(&at_ref, 40.0, 20.0).unwrap()
        );
        for t in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(zero_field_resistivity(&base, &[t]).is_err());
            assert!(hall_coefficient(&base, t, false).is_err());
            assert!(magnetoresistance(&base, &[t], 35.0, 0.5).is_err());
        }
        assert!(conductivity_tensor(&base, 40.0, &[f64::NAN]).is_err());
        assert!(hall_coefficient_chambers(&base, 40.0, 0.0).is_err());
        for (field_max, step) in [(0.0, 0.5), (35.0, 0.0), (f64::NAN, 0.5), (35.0, 1e-300)] {
            assert!(field_grid(field_max, step).is_err(), "{field_max} {step}");
        }
        assert!(magnetoresistance(&base, &[40.0], 0.5, 0.5).is_err());
        assert!(fit_resistivity_polynomial(&[1.0, 2.0], &[1.0, 2.0]).is_err());
        assert!(fit_resistivity_polynomial(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
        assert!(fit_resistivity_polynomial(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).is_err());
        assert!(resistivity_fit(&base, 60.0, 15.0, 1.0).is_err());
        // Glue and pairing.
        let glue = SfGlueConfig::default();
        let bad_pairing = SfPairingMap {
            lambda_ref: SF_POWER_LAW_LAMBDA_C,
            ..Default::default()
        };
        let glue_cases = [
            SfGlueConfig { eta: -0.1, ..glue },
            SfGlueConfig { eta: 1.0, ..glue },
            SfGlueConfig {
                eta: f64::NAN,
                ..glue
            },
            SfGlueConfig {
                tc_target: 0.0,
                ..glue
            },
            SfGlueConfig {
                tc_target: f64::NAN,
                ..glue
            },
            SfGlueConfig {
                tc_target: f64::INFINITY,
                ..glue
            },
            SfGlueConfig {
                tc_target: 50.0,
                pairing: mcmillan(),
                ..glue
            },
            SfGlueConfig {
                delta_avg: 0.0,
                ..glue
            },
            SfGlueConfig {
                delta_avg: f64::NAN,
                ..glue
            },
            SfGlueConfig {
                moire_period: 0.0,
                ..glue
            },
            SfGlueConfig {
                moire_period: -1.0,
                ..glue
            },
            SfGlueConfig {
                moire_period: f64::INFINITY,
                ..glue
            },
            SfGlueConfig {
                moire_period: f64::NAN,
                ..glue
            },
            SfGlueConfig {
                pairing: bad_pairing,
                ..glue
            },
            SfGlueConfig {
                pairing: SfPairingMap {
                    exponent: 0.0,
                    ..Default::default()
                },
                ..glue
            },
            SfGlueConfig {
                pairing: SfPairingMap {
                    mu_star: 2.0,
                    ..mcmillan()
                },
                ..glue
            },
        ];
        for g in &glue_cases {
            assert!(
                glue_modulated_gap(&[0.0, 0.5, 1.0], g).is_err(),
                "glue = {g:?}"
            );
            assert!(coherence_factor(g).is_err(), "glue = {g:?}");
            assert!(linearized_glue_amplitude(g).is_err(), "glue = {g:?}");
        }
        assert!(glue_modulated_gap(&[], &glue).is_err());
        assert!(glue_modulated_gap(&[0.5, f64::NAN], &glue).is_err());
        for tc in [0.0, -1.0, f64::NAN] {
            assert!(imprint_curve(&glue, &[1.0, tc]).is_err());
        }
        assert!(imprint_curve(
            &SfGlueConfig {
                pairing: mcmillan(),
                ..glue
            },
            &[50.0]
        )
        .is_err());
        assert!(local_imprint(1.0, 13.5, &glue.pairing).is_err());
        assert!(local_imprint(0.1, 0.0, &glue.pairing).is_err());
        assert!(lambda_from_tc(13.5, &bad_pairing).is_err());
        assert!(fragmentation_tc(&SfGlueConfig { eta: 1.0, ..glue }).is_err());
    }

    // --- Cross-stack parity anchors v2 (pinned identically in
    // tests/test_spin_fluctuation.py; SF-amendments §9.3; AP = algorithm parity) ---

    #[test]
    fn test_parity_fermi_surface() {
        let c = SfConfig::default();
        let (kf, vf, gamma, _, _) = fermi_surface_at(0.3, &c);
        assert_close(kf, 5614221434.684519, 1e-14, "P1 k_F");
        assert_close(vf, 499330.6678195376, 1e-14, "P1 v_F");
        assert_close(gamma, 0.19556905668726163, 1e-14, "P1 gamma");
        assert_close(
            luttinger_holes(&c).unwrap(),
            0.8391564248900439,
            1e-12,
            "P1b",
        );
        assert_close(
            zone_boundary_crossing_deg(&c).unwrap(),
            27.196569908994878,
            1e-12,
            "P1c",
        );
        assert_close(
            drude_hall_coefficient(&c).unwrap(),
            7.403312411077447e-10,
            1e-12,
            "P1d",
        );
    }

    #[test]
    fn test_parity_scattering_rate() {
        let c = SfConfig::default();
        let at = |cfg: &SfConfig, deg: f64| scattering_rate(deg.to_radians(), 40.0, cfg).unwrap();
        for (deg, want) in [
            (27.5, 106.3201227826564),
            (45.0, 35.74824703294329),
            (0.0, 27.12151850594606),
        ] {
            assert_close(at(&c, deg), want, 1e-13, "P2");
        }
        let antinodal = SfConfig {
            inelastic: SfInelasticForm::Antinodal,
            ..c
        };
        let isotropic = SfConfig {
            inelastic: SfInelasticForm::Isotropic,
            ..c
        };
        let degs = [0.0, 10.0, 27.5];
        for (deg, want) in degs
            .iter()
            .zip([74.92, 69.31442485713072, 74.16519736591698])
        {
            assert_close(at(&antinodal, *deg), want, 1e-13, "P2b antinodal");
        }
        for (deg, want) in degs.iter().zip([74.92, 74.92, 106.32]) {
            assert_close(at(&isotropic, *deg), want, 1e-13, "P2b isotropic");
        }
    }

    #[test]
    fn test_parity_zero_field_rho() {
        let rho = zero_field_resistivity(&SfConfig::default(), &[40.0]).unwrap();
        assert_close(rho[0], 24.49517992890017, 1e-12, "P3");
    }

    #[test]
    fn test_parity_mr_20t() {
        // P4 (AP).
        let r = resistivity_tensor(&SfConfig::default(), 40.0, &[0.0, 20.0]).unwrap();
        assert_close(r[1][0], 25.04265480287949, 1e-12, "P4 rho_xx");
        assert_close(r[1][1], 1.192527321423354, 1e-12, "P4 rho_yx");
        assert_close(
            r[1][0] / r[0][0] - 1.0,
            0.022350310369975812,
            1e-11,
            "P4 MR",
        );
    }

    #[test]
    fn test_parity_mr_35t_20k() {
        // P4b (AP), through the MR family (one temperature, 0-35 T).
        let fam =
            magnetoresistance(&SfConfig::default(), &[20.0], SF_FIELD_MAX, SF_FIELD_STEP).unwrap();
        assert_close(fam.mr[0][70], 0.041376414577780185, 1e-11, "P4b");
    }

    #[test]
    fn test_parity_high_field_slope() {
        // P5 (AP).
        let slope =
            high_field_slope(&SfConfig::default(), 40.0, SF_SLOPE_FIELD, SF_FIELD_STEP).unwrap();
        assert_close(slope, 0.05026085857440421, 1e-9, "P5");
    }

    #[test]
    fn test_parity_hall() {
        let c = SfConfig::default();
        let a = case_a(&c);
        assert_close(rh(&a, 0.0, false), 7.339685060373182e-10, 1e-12, "P6 iso");
        assert_close(
            rh(&a, 0.0, true),
            5.309220240734655e-10,
            1e-12,
            "P6 iso CVC",
        );
        assert_close(rh(&c, 40.0, false), 6.131175356207996e-10, 1e-12, "P6 40 K");
        assert_close(
            rh(&c, 40.0, true),
            5.092606357247605e-10,
            1e-12,
            "P6 40 K CVC",
        );
        // P6b (AP).
        let chambers = hall_coefficient_chambers(&c, 40.0, 0.1).unwrap();
        assert_close(chambers, 6.131011216626595e-10, 1e-11, "P6b");
        let antinodal = SfConfig {
            inelastic: SfInelasticForm::Antinodal,
            ..c
        };
        let none_hs = SfConfig {
            inelastic: SfInelasticForm::None,
            ..c
        };
        assert_close(
            rh(&antinodal, 50.0, true),
            1.9749077940876073e-09,
            1e-12,
            "P6c antinodal",
        );
        assert_close(
            rh(&none_hs, 0.0, true),
            4.928676220661514e-10,
            1e-12,
            "P6c none + HS",
        );
        let centres = cvc_centres(&c).unwrap();
        assert_close(
            centres[0].to_degrees(),
            27.196569908994878,
            1e-12,
            "P6d centre 1",
        );
        assert_close(
            centres[1].to_degrees(),
            62.80343009100512,
            1e-12,
            "P6d centre 2",
        );
        assert!(
            (cvc_partner_angle(0.0, &c) - PI).abs() <= 1e-15,
            "P6d partner"
        );
        assert_close(
            hall_ratio_to_case_a(&c, 40.0).unwrap(),
            0.9592004336484117,
            1e-12,
            "P6e",
        );
    }

    #[test]
    fn test_parity_lambda_tr() {
        let c = SfConfig::default();
        assert_close(
            lambda_tr(&c).unwrap(),
            0.41758273992325223,
            1e-12,
            "P7 0.159",
        );
        let x116 = SfConfig { doping: 0.116, ..c };
        assert_close(
            lambda_tr(&x116).unwrap(),
            2.714287809501139,
            1e-12,
            "P7 0.116",
        );
        let iso = SfConfig {
            inelastic: SfInelasticForm::Isotropic,
            ..c
        };
        let antinodal = SfConfig {
            inelastic: SfInelasticForm::Antinodal,
            ..c
        };
        assert_close(
            lambda_tr(&iso).unwrap(),
            0.9117468534052734,
            1e-12,
            "P7b isotropic",
        );
        assert_close(
            lambda_tr(&antinodal).unwrap(),
            0.4475272735616924,
            1e-12,
            "P7b antinodal",
        );
    }

    #[test]
    fn test_parity_lambda_rho() {
        let c = SfConfig::default();
        assert_close(
            drude_resistivity_per_rate(&c).unwrap(),
            5.5005672382590704e-21,
            1e-12,
            "P7c rho_D",
        );
        for (doping, want) in [
            (0.159, 0.4352438971760371),
            (0.116, 1.491881542660157),
            (0.188, 0.2946327374251149),
        ] {
            let cx = SfConfig { doping, ..c };
            assert_close(lambda_resistive(&cx).unwrap(), want, 1e-9, "P7c lambda_rho");
        }
    }

    #[test]
    fn test_parity_pairing() {
        let pl = SfPairingMap::default();
        let mc = mcmillan();
        assert_close(pl.lambda_c, 0.30462015776911294, 1e-12, "P8 lambda_c");
        assert_close(pl.exponent, 0.3926512381199364, 1e-12, "P8 p");
        assert_close(pl.lambda_ref, 2.1714302476009117, 1e-12, "P8 lambda_ref");
        assert_close(mc.omega_k, 178.04201155099287, 1e-12, "P8 omega");
        assert_eq!(mc.mu_star, 0.0);
        let lam116 = lambda_tr(&SfConfig {
            doping: 0.116,
            ..Default::default()
        })
        .unwrap();
        assert_close(
            tc_from_lambda(lam116, &pl),
            27.635552408315235,
            1e-12,
            "P8 power law",
        );
        assert_close(
            tc_from_lambda(lam116, &mc),
            29.58591451200863,
            1e-12,
            "P8 McMillan",
        );
        // P9 and P9b.
        let lb = lambda_from_tc(13.5, &pl).unwrap();
        assert_close(lb, 0.6932743989110509, 1e-12, "P9 lambda_bar");
        assert_close(log_sensitivity(lb, &pl), 0.700404169756287, 1e-12, "P9 S");
        let lbm = lambda_from_tc(13.5, &mc).unwrap();
        assert_close(lbm, 0.8905891525900327, 1e-12, "P9 McMillan lambda_bar");
        assert_close(
            log_sensitivity(lbm, &mc),
            1.16776630051629,
            1e-12,
            "P9 McMillan S",
        );
        assert_close(
            equivalent_doping(lb).unwrap(),
            0.1435584333754864,
            1e-12,
            "P9b x_eq",
        );
        assert_close(
            equivalent_doping(pl.lambda_c).unwrap(),
            0.17052577242235883,
            1e-12,
            "P9b x_c",
        );
    }

    #[test]
    fn test_parity_glue() {
        // P10: Delta at m = 0, 1/2, 1 (meV), eta = 0.1; the 0.0 entries are exact.
        let gaps = [
            (
                13.5,
                SfCoherence::Clean,
                [2.9569962300271597, 3.09, 3.2092567693397047],
            ),
            (
                13.5,
                SfCoherence::Dirty,
                [2.9569962300271597, 3.09, 3.2092567693397047],
            ),
            (
                13.5,
                SfCoherence::Local,
                [2.860581453791304, 3.09, 3.295706309325275],
            ),
            (
                5.0,
                SfCoherence::Clean,
                [0.8818228883664684, 1.1444444444444444, 1.2321772587067223],
            ),
            (
                5.0,
                SfCoherence::Dirty,
                [0.6771928153962904, 1.1444444444444444, 1.300537116337901],
            ),
            (
                5.0,
                SfCoherence::Local,
                [0.0, 1.1444444444444444, 1.5267639058679714],
            ),
            (
                2.0,
                SfCoherence::Clean,
                [0.4462301216137267, 0.45777777777777773, 0.4760946886571619],
            ),
            (
                2.0,
                SfCoherence::Dirty,
                [0.34672682120871845, 0.45777777777777773, 0.6339269846513088],
            ),
            (
                2.0,
                SfCoherence::Local,
                [0.0, 0.45777777777777773, 1.1839056290920276],
            ),
        ];
        for (tc_target, coherence, want) in gaps {
            let glue = SfGlueConfig {
                tc_target,
                coherence,
                ..Default::default()
            };
            let res = glue_modulated_gap(&[0.0, 0.5, 1.0], &glue).unwrap();
            for (&v, w) in res.gap.iter().zip(want) {
                if w == 0.0 {
                    assert_eq!(v, 0.0, "P10 {tc_target} K {}", coherence.as_str());
                } else {
                    assert_close(v, w, 1e-12, "P10");
                }
            }
        }
        // P10b: xi clean, xi dirty, F clean, F dirty, Delta_bar.
        let scales = [
            (
                13.5,
                [20.0, 20.0, 0.5797428855287498, 0.5797428855287498, 3.09],
            ),
            (
                5.0,
                [
                    54.0,
                    32.86335345030997,
                    0.22947514608755187,
                    0.40827812246926076,
                    1.1444444444444444,
                ],
            ),
            (
                2.0,
                [
                    135.0,
                    51.96152422706632,
                    0.02522546249428618,
                    0.24258704104891599,
                    0.45777777777777773,
                ],
            ),
        ];
        for (tc_target, want) in scales {
            let clean = SfGlueConfig {
                tc_target,
                ..Default::default()
            };
            let dirty = SfGlueConfig {
                coherence: SfCoherence::Dirty,
                ..clean
            };
            let got = [
                coherence_length(&clean).unwrap(),
                coherence_length(&dirty).unwrap(),
                coherence_factor(&clean).unwrap(),
                coherence_factor(&dirty).unwrap(),
                glue_gap_scale(&clean).unwrap(),
            ];
            for (v, w) in got.iter().zip(want) {
                assert_close(*v, w, 1e-14, "P10b");
            }
        }
        // P10c: dressed clean, dressed dirty, local exact, local linear.
        let imprints = [
            (
                13.5,
                [
                    0.04081885749393931,
                    0.04081885749393931,
                    0.0704085526753999,
                    0.07004041697562871,
                ],
            ),
            (
                5.0,
                [
                    0.1530674433525382,
                    0.27233488876089784,
                    0.6670327744083371,
                    0.4254496324492375,
                ],
            ),
            (
                2.0,
                [
                    0.03261906594549964,
                    0.313689498905742,
                    1.293100808474302,
                    4.022759727552418,
                ],
            ),
        ];
        for (tc_target, want) in imprints {
            let (clean, local, linear) =
                imprint_curve(&SfGlueConfig::default(), &[tc_target]).unwrap();
            let dirty_glue = SfGlueConfig {
                coherence: SfCoherence::Dirty,
                ..Default::default()
            };
            let (dirty, _, _) = imprint_curve(&dirty_glue, &[tc_target]).unwrap();
            for (v, w) in [clean[0], dirty[0], local[0], linear[0]].iter().zip(want) {
                assert_close(*v, w, 1e-12, "P10c");
            }
        }
        // P10d-f.
        assert_close(
            SfGlueConfig::default().moire_period,
            36.68576576576573,
            1e-15,
            "P10d",
        );
        assert_close(
            fragmentation_tc(&SfGlueConfig::default()).unwrap(),
            5.177318594015492,
            1e-12,
            "P10e",
        );
        let amplitude = linearized_glue_amplitude(&SfGlueConfig::default()).unwrap();
        assert_close(amplitude, 0.12547078933296135, 1e-12, "P10f");
    }

    #[test]
    fn test_parity_quadrature() {
        let q = SfQuadratureConfig::default();
        assert_close(
            quadrature_field_derivative(40.0, 20.0, &q),
            0.18569533817705186,
            1e-13,
            "P11 slope",
        );
        assert_close(
            quadrature_resistivity(40.0, 20.0, &q),
            30.208131845707605,
            1e-13,
            "P11 rho",
        );
    }

    #[test]
    fn test_parity_rho_fit() {
        let (t_min, t_max) = SF_RHO_FIT_RANGE;
        let fit = resistivity_fit(&SfConfig::default(), t_min, t_max, 1.0).unwrap();
        assert_close(fit.rho0, 15.80541402181645, 1e-9, "P12 rho0");
        assert_close(fit.a1, 0.19693692765571835, 1e-9, "P12 A1");
        assert_close(fit.a2, 0.0005046269609317277, 1e-9, "P12 A2");
        assert_eq!((fit.t_min, fit.t_max), (15.0, 60.0));
    }

    #[test]
    fn test_parity_doping_0116() {
        let x116 = SfConfig {
            doping: 0.116,
            ..Default::default()
        };
        let r = resistivity_tensor(&x116, 40.0, &[0.0, 20.0]).unwrap();
        assert_close(r[0][0], 41.96086725955268, 1e-12, "P13 rho(0)");
        assert_close(
            r[1][0] / r[0][0] - 1.0,
            0.03951909517654206,
            1e-11,
            "P13 MR (AP)",
        );
    }

    // --- Accuracy against converged values (SF-amendments §9.3b) ---

    #[test]
    fn test_accuracy_converged() {
        let c = SfConfig::default();
        let r40 = resistivity_tensor(&c, 40.0, &[0.0, 20.0]).unwrap();
        let r20 = resistivity_tensor(&c, 20.0, &[0.0, 35.0]).unwrap();
        let x116 = SfConfig { doping: 0.116, ..c };
        let r116 = resistivity_tensor(&x116, 40.0, &[0.0, 20.0]).unwrap();
        let checks = [
            (r40[1][0], 25.04281275669245, "rho_xx(20 T, 40 K)"),
            (r40[1][1], 1.1926091496601852, "rho_yx(20 T, 40 K)"),
            (
                r40[1][0] / r40[0][0] - 1.0,
                0.022356758733017124,
                "MR(20 T, 40 K)",
            ),
            (
                r20[1][0] / r20[0][0] - 1.0,
                0.04139795183670203,
                "MR(35 T, 20 K)",
            ),
            (
                high_field_slope(&c, 40.0, SF_SLOPE_FIELD, SF_FIELD_STEP).unwrap(),
                0.05027355287307245,
                "gamma_1(33 T, 40 K)",
            ),
            (
                r116[1][0] / r116[0][0] - 1.0,
                0.03953104183231093,
                "MR(20 T, 40 K), x = 0.116",
            ),
            (
                hall_coefficient_chambers(&c, 40.0, 0.1).unwrap(),
                6.131167143845524e-10,
                "R_H Chambers 0.1 T",
            ),
        ];
        for (v, want, what) in checks {
            assert_close(v, want, 1e-3, what);
        }
    }
}
