//! Speculative isotope-effect calculations for moire CPDM systems.
//!
//! **SPECULATIVE** — These calculations use simplified models (zero-point
//! lattice expansion, BCS isotope effect, Debye-Waller factor) to estimate
//! how isotopic substitution might affect moire patterns and superconducting
//! gap modulation.  Results are qualitative.
//!
//! Literature basis for the BCS isotope exponent (α):
//!   - FeSe: α_Fe = 0.81 ± 0.15 (Khasanov et al., arXiv:1002.2510)
//!   - SmFeAsO: α_Fe ≈ 0.35 (Liu et al., Nature 459, 64, 2009)
//!   - (Ba,K)Fe₂As₂: α_Fe = −0.18 — inverse effect (Shirage, PRL 103, 257003)
//!   - Corrected consensus: α ≈ 0.35–0.4 (PRB 82, 212505)
//!   - No Te isotope data for FeTe; key experimental gap.

use crate::isotopes::{
    element_by_symbol, formula_unit_avg_mass, material_composition, te_125_spin_fraction,
    IsotopeConfig,
};
use crate::materials::LatticeType;
use std::f64::consts::PI;

/// Physical constants.
const KB_EV_K: f64 = 8.617333262e-5; // Boltzmann constant in eV/K
const AMU_TO_KG: f64 = 1.66053906660e-27;
const HBAR_J_S: f64 = 1.054571817e-34;
const KB_J_K: f64 = 1.380649e-23; // Boltzmann constant in J/K

/// Default superconducting parameters.
const DELTA_1_DEFAULT: f64 = 2.58; // meV
const DELTA_2_DEFAULT: f64 = 3.60; // meV
const DELTA_AVG_DEFAULT: f64 = (DELTA_1_DEFAULT + DELTA_2_DEFAULT) / 2.0;
const COHERENCE_LENGTH_DEFAULT: f64 = 20.0; // Angstrom
// Literature range: α = 0.35–0.8 for iron chalcogenides (Khasanov 2010, Liu 2009);
// α = −0.18 inverse effect in (Ba,K)Fe₂As₂ (Shirage 2009).
// 0.4 is the corrected consensus value (PRB 82, 212505).
const ISOTOPE_EXPONENT_DEFAULT: f64 = 0.4;

/// Configuration for isotope-effect computation.
#[derive(Debug, Clone)]
pub struct IsotopeEffectsConfig {
    pub substrate_formula: &'static str,
    pub overlayer_formula: &'static str,
    pub substrate_a: f64,
    pub overlayer_a: f64,
    pub substrate_lattice_type: LatticeType,
    pub overlayer_lattice_type: LatticeType,
    pub delta_1: f64,
    pub delta_2: f64,
    pub coherence_length: f64,
    pub isotope_config: IsotopeConfig,
    pub alpha: f64,
}

impl Default for IsotopeEffectsConfig {
    fn default() -> Self {
        Self {
            substrate_formula: "FeTe",
            overlayer_formula: "Sb2Te3",
            substrate_a: 3.82,
            overlayer_a: 4.264,
            substrate_lattice_type: LatticeType::Square,
            overlayer_lattice_type: LatticeType::Hexagonal,
            delta_1: DELTA_1_DEFAULT,
            delta_2: DELTA_2_DEFAULT,
            coherence_length: COHERENCE_LENGTH_DEFAULT,
            isotope_config: IsotopeConfig::default(),
            alpha: ISOTOPE_EXPONENT_DEFAULT,
        }
    }
}

/// All computed isotope modifications.
#[derive(Debug, Clone)]
pub struct IsotopeEffects {
    pub substrate_a_modified: f64,
    pub substrate_delta_a: f64,
    pub overlayer_a_modified: f64,
    pub overlayer_delta_a: f64,
    pub delta_1_modified: f64,
    pub delta_2_modified: f64,
    pub coherence_length_modified: f64,
    pub dw_factor_substrate: f64,
    pub dw_factor_overlayer: f64,
    /// Isotope-shifted substrate Debye temperature (K): Θ_nat·√(M_nat/M_enr).
    pub theta_d_substrate: f64,
    /// Isotope-shifted overlayer Debye temperature (K).
    pub theta_d_overlayer: f64,
    /// Fraction of Te that is ¹²⁵Te (I=1/2, the only spin-bearing Te isotope).
    pub te_125_spin_fraction: f64,
}

/// Compute stoichiometry-weighted bulk property for a material formula.
fn weighted_property(formula: &str, prop_fn: fn(&str) -> f64) -> f64 {
    let comp = match material_composition(formula) {
        Some(c) => c,
        None => return 0.0,
    };
    let total_atoms: u32 = comp.iter().map(|(_, n)| n).sum();
    if total_atoms == 0 {
        return 0.0;
    }
    let total: f64 = comp.iter().map(|(sym, n)| *n as f64 * prop_fn(sym)).sum();
    total / total_atoms as f64
}

fn elem_debye(sym: &str) -> f64 {
    element_by_symbol(sym).map_or(0.0, |e| e.debye_temperature)
}

fn elem_gruneisen(sym: &str) -> f64 {
    element_by_symbol(sym).map_or(0.0, |e| e.gruneisen_parameter)
}

fn elem_cohesive(sym: &str) -> f64 {
    element_by_symbol(sym).map_or(0.0, |e| e.cohesive_energy_ev)
}

/// Zero-point lattice constant shift.
///
/// Heavier enrichment ⇒ contraction via reduced zero-point anharmonic
/// expansion. The prefactor uses the natural Θ_D (leading order).
fn lattice_shift(formula: &str, base_a: f64, config: &IsotopeConfig) -> (f64, f64) {
    let natural_config = IsotopeConfig::default();
    let m_natural = match formula_unit_avg_mass(formula, &natural_config) {
        Some(m) => m,
        None => return (base_a, 0.0),
    };
    let m_enriched = match formula_unit_avg_mass(formula, config) {
        Some(m) => m,
        None => return (base_a, 0.0),
    };

    if m_enriched <= 0.0 || m_natural <= 0.0 {
        return (base_a, 0.0);
    }

    let gamma_g = weighted_property(formula, elem_gruneisen);
    let t_debye = weighted_property(formula, elem_debye);
    let e_coh = weighted_property(formula, elem_cohesive);

    if e_coh <= 0.0 {
        return (base_a, 0.0);
    }

    let prefactor = 3.0 * gamma_g * KB_EV_K * t_debye / (4.0 * e_coh);
    let mass_term = 1.0 - (m_natural / m_enriched).sqrt();
    let delta_a = -base_a * prefactor * mass_term;

    (base_a + delta_a, delta_a)
}

/// BCS isotope effect on superconducting gap: Δ ∝ (M_nat/M_enr)^α.
fn gap_modification(
    delta_1: f64,
    delta_2: f64,
    substrate_formula: &str,
    config: &IsotopeConfig,
    alpha: f64,
) -> (f64, f64) {
    let natural_config = IsotopeConfig::default();
    let m_natural = match formula_unit_avg_mass(substrate_formula, &natural_config) {
        Some(m) => m,
        None => return (delta_1, delta_2),
    };
    let m_enriched = match formula_unit_avg_mass(substrate_formula, config) {
        Some(m) => m,
        None => return (delta_1, delta_2),
    };

    if m_enriched <= 0.0 || m_natural <= 0.0 {
        return (delta_1, delta_2);
    }

    let ratio = (m_natural / m_enriched).powf(alpha);
    (delta_1 * ratio, delta_2 * ratio)
}

/// Modified coherence length from gap change.
fn coherence_modification(xi_0: f64, delta_avg_orig: f64, d1_mod: f64, d2_mod: f64) -> f64 {
    let delta_avg_mod = (d1_mod + d2_mod) / 2.0;
    if delta_avg_mod <= 0.0 {
        return xi_0;
    }
    xi_0 * (delta_avg_orig / delta_avg_mod)
}

/// Debye-Waller factor ratio (enriched vs natural), zero-point model.
///
/// Zero-point mean-square displacement ⟨u²⟩_zp = 9ħ²/(4·M·k_B·Θ_D(M)) with
/// the Debye temperature co-varying with isotope mass,
/// Θ_D(M) = Θ_nat·√(M_nat/M). With the isotropic projection ⟨u_G²⟩ = ⟨u²⟩/3
/// the per-component exponent is
///   2W(M) = G² · (3ħ²/(4 k_B Θ_nat)) · 1/√(M·M_nat),
/// so the enriched/natural ratio is
///   DW_ratio = exp(−G²·C·(1/√(M_enr·M_nat) − 1/M_nat)),
///   C = 3ħ²/(4 k_B Θ_nat), all SI (masses in kg).
///
/// Zero-point only — and that is the right model: the classical (high-T)
/// thermal MSD 3k_BT/(Mω̄²) is mass-independent because Mω̄² does not depend
/// on isotope mass, so the isotope contrast in the DW factor is purely a
/// quantum zero-point effect. Heavier enrichment ⇒ ratio > 1 (less
/// zero-point smearing, sharper potential); lighter ⇒ < 1.
fn debye_waller_ratio(
    formula: &str,
    lattice_type: LatticeType,
    a: f64,
    config: &IsotopeConfig,
) -> f64 {
    let natural_config = IsotopeConfig::default();
    let m_natural = match formula_unit_avg_mass(formula, &natural_config) {
        Some(m) => m,
        None => return 1.0,
    };
    let m_enriched = match formula_unit_avg_mass(formula, config) {
        Some(m) => m,
        None => return 1.0,
    };

    if m_enriched <= 0.0 || m_natural <= 0.0 {
        return 1.0;
    }

    let t_debye_nat = weighted_property(formula, elem_debye);
    if t_debye_nat <= 0.0 {
        return 1.0;
    }
    // C = 3*hbar^2 / (4 * k_B * Theta_nat)
    let c_val = 3.0 * HBAR_J_S * HBAR_J_S / (4.0 * KB_J_K * t_debye_nat);

    // G magnitude in 1/m
    let ang_to_m = 1e-10;
    let g_mag = match lattice_type {
        LatticeType::Hexagonal => 4.0 * PI / (a * 3.0_f64.sqrt()),
        LatticeType::Square => 2.0 * PI / a,
    };
    let g_si = g_mag / ang_to_m;

    let m_nat_kg = m_natural * AMU_TO_KG;
    let m_enr_kg = m_enriched * AMU_TO_KG;
    let inv_mass_term = 1.0 / (m_enr_kg * m_nat_kg).sqrt() - 1.0 / m_nat_kg;

    let exponent = (-g_si * g_si * c_val * inv_mass_term).clamp(-100.0, 100.0);
    exponent.exp()
}

/// Isotope-shifted Debye temperature (K): Θ_enr = Θ_nat·√(M_nat/M_enr).
///
/// Θ_nat is the stoichiometry-weighted average Debye temperature and the M's
/// are the formula-unit per-atom average masses.
fn isotope_debye_temperature(formula: &str, config: &IsotopeConfig) -> f64 {
    let t_debye_nat = weighted_property(formula, elem_debye);
    let natural_config = IsotopeConfig::default();
    let m_natural = formula_unit_avg_mass(formula, &natural_config);
    let m_enriched = formula_unit_avg_mass(formula, config);
    match (m_natural, m_enriched) {
        (Some(m_nat), Some(m_enr)) if m_nat > 0.0 && m_enr > 0.0 => {
            t_debye_nat * (m_nat / m_enr).sqrt()
        }
        _ => t_debye_nat,
    }
}

/// Compute all speculative isotope effects for a substrate/overlayer pair.
pub fn compute_isotope_effects(cfg: &IsotopeEffectsConfig) -> IsotopeEffects {
    let (sub_a_mod, sub_da) =
        lattice_shift(cfg.substrate_formula, cfg.substrate_a, &cfg.isotope_config);
    let (over_a_mod, over_da) =
        lattice_shift(cfg.overlayer_formula, cfg.overlayer_a, &cfg.isotope_config);

    let (d1_mod, d2_mod) = gap_modification(
        cfg.delta_1,
        cfg.delta_2,
        cfg.substrate_formula,
        &cfg.isotope_config,
        cfg.alpha,
    );

    let xi_mod = coherence_modification(cfg.coherence_length, DELTA_AVG_DEFAULT, d1_mod, d2_mod);

    let dw_sub = debye_waller_ratio(
        cfg.substrate_formula,
        cfg.substrate_lattice_type,
        cfg.substrate_a,
        &cfg.isotope_config,
    );
    let dw_over = debye_waller_ratio(
        cfg.overlayer_formula,
        cfg.overlayer_lattice_type,
        cfg.overlayer_a,
        &cfg.isotope_config,
    );

    let theta_d_sub = isotope_debye_temperature(cfg.substrate_formula, &cfg.isotope_config);
    let theta_d_over = isotope_debye_temperature(cfg.overlayer_formula, &cfg.isotope_config);

    let spin_frac = te_125_spin_fraction(cfg.isotope_config.te_mass);

    IsotopeEffects {
        substrate_a_modified: sub_a_mod,
        substrate_delta_a: sub_da,
        overlayer_a_modified: over_a_mod,
        overlayer_delta_a: over_da,
        delta_1_modified: d1_mod,
        delta_2_modified: d2_mod,
        coherence_length_modified: xi_mod,
        dw_factor_substrate: dw_sub,
        dw_factor_overlayer: dw_over,
        theta_d_substrate: theta_d_sub,
        theta_d_overlayer: theta_d_over,
        te_125_spin_fraction: spin_frac,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_natural_mass_gives_zero_shift() {
        let cfg = IsotopeEffectsConfig::default();
        let effects = compute_isotope_effects(&cfg);
        assert!(
            effects.substrate_delta_a.abs() < 1e-12,
            "Expected zero shift with natural masses, got {}",
            effects.substrate_delta_a
        );
        assert!(effects.overlayer_delta_a.abs() < 1e-12);
    }

    #[test]
    fn test_natural_mass_gives_identity_gap() {
        let cfg = IsotopeEffectsConfig::default();
        let effects = compute_isotope_effects(&cfg);
        assert!((effects.delta_1_modified - DELTA_1_DEFAULT).abs() < 1e-12);
        assert!((effects.delta_2_modified - DELTA_2_DEFAULT).abs() < 1e-12);
    }

    #[test]
    fn test_natural_mass_gives_identity_dw() {
        let cfg = IsotopeEffectsConfig::default();
        let effects = compute_isotope_effects(&cfg);
        assert!((effects.dw_factor_substrate - 1.0).abs() < 1e-10);
        assert!((effects.dw_factor_overlayer - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_heavier_isotope_contracts_lattice() {
        // Enriching Fe to 58 (heavier) should give negative delta_a
        let cfg = IsotopeEffectsConfig {
            isotope_config: IsotopeConfig {
                fe_mass: Some(57.9333), // Fe-58
                ..Default::default()
            },
            ..Default::default()
        };
        let effects = compute_isotope_effects(&cfg);
        assert!(
            effects.substrate_delta_a < 0.0,
            "Heavier isotope should contract lattice, got delta_a = {}",
            effects.substrate_delta_a
        );
    }

    #[test]
    fn test_lighter_isotope_expands_lattice() {
        let cfg = IsotopeEffectsConfig {
            isotope_config: IsotopeConfig {
                fe_mass: Some(53.9396), // Fe-54
                ..Default::default()
            },
            ..Default::default()
        };
        let effects = compute_isotope_effects(&cfg);
        assert!(
            effects.substrate_delta_a > 0.0,
            "Lighter isotope should expand lattice, got delta_a = {}",
            effects.substrate_delta_a
        );
    }

    #[test]
    fn test_lighter_isotope_increases_gap() {
        // Lighter mass -> higher Debye frequency -> larger gap (positive alpha)
        let cfg = IsotopeEffectsConfig {
            isotope_config: IsotopeConfig {
                fe_mass: Some(53.9396),
                ..Default::default()
            },
            ..Default::default()
        };
        let effects = compute_isotope_effects(&cfg);
        assert!(effects.delta_1_modified > DELTA_1_DEFAULT);
        assert!(effects.delta_2_modified > DELTA_2_DEFAULT);
    }

    #[test]
    fn test_lattice_shift_magnitude() {
        // Shift should be small: order of 1e-4 to 1e-2 Angstrom
        let cfg = IsotopeEffectsConfig {
            isotope_config: IsotopeConfig {
                fe_mass: Some(53.9396),
                ..Default::default()
            },
            ..Default::default()
        };
        let effects = compute_isotope_effects(&cfg);
        let abs_da = effects.substrate_delta_a.abs();
        assert!(
            abs_da > 1e-6 && abs_da < 0.1,
            "Lattice shift magnitude {} outside expected range",
            abs_da
        );
    }

    #[test]
    fn test_larger_gap_shortens_coherence() {
        let cfg = IsotopeEffectsConfig {
            isotope_config: IsotopeConfig {
                fe_mass: Some(53.9396),
                ..Default::default()
            },
            ..Default::default()
        };
        let effects = compute_isotope_effects(&cfg);
        assert!(effects.coherence_length_modified < COHERENCE_LENGTH_DEFAULT);
    }

    /// Default config (FeTe substrate) with a single mass override.
    fn effects_with(isotope_config: IsotopeConfig) -> IsotopeEffects {
        compute_isotope_effects(&IsotopeEffectsConfig {
            isotope_config,
            ..Default::default()
        })
    }

    // --- Debye-Waller (zero-point model) ---

    #[test]
    fn test_dw_heavier_enrichment_above_unity() {
        // Pure 130Te (heavier than natural) ⇒ less zero-point smearing ⇒ > 1.
        let effects = effects_with(IsotopeConfig {
            te_mass: Some(129.906),
            ..Default::default()
        });
        assert!(
            effects.dw_factor_substrate > 1.0,
            "heavier enrichment must give DW ratio > 1, got {}",
            effects.dw_factor_substrate
        );
    }

    #[test]
    fn test_dw_lighter_enrichment_below_unity() {
        // Pure 54Fe (lighter than natural) ⇒ more zero-point smearing ⇒ < 1.
        let effects = effects_with(IsotopeConfig {
            fe_mass: Some(53.9396),
            ..Default::default()
        });
        assert!(
            effects.dw_factor_substrate < 1.0,
            "lighter enrichment must give DW ratio < 1, got {}",
            effects.dw_factor_substrate
        );
    }

    #[test]
    fn test_dw_parity_anchor_130te_fete() {
        // Cross-language parity anchor: FeTe (a = 3.82, square), pure 130Te,
        // natural Fe. M_nat = 91.235683 amu, M_enr = 92.876590 amu,
        // Θ_nat = 212.5 K, G = 2π/3.82 Å⁻¹ ⇒ DW ratio = 1.0000450.
        let effects = effects_with(IsotopeConfig {
            te_mass: Some(129.906),
            ..Default::default()
        });
        assert!(
            (effects.dw_factor_substrate - 1.0000450).abs() < 5e-6,
            "DW parity anchor mismatch: got {}",
            effects.dw_factor_substrate
        );
    }

    // --- Isotope-shifted Debye temperature ---

    #[test]
    fn test_theta_d_natural() {
        let effects = compute_isotope_effects(&IsotopeEffectsConfig::default());
        // Natural FeTe: (260 + 165)/2 = 212.5 K exactly.
        assert!((effects.theta_d_substrate - 212.5).abs() < 1e-9);
        // Natural Sb2Te3: (2·210 + 3·165)/5 = 183.0 K exactly.
        assert!((effects.theta_d_overlayer - 183.0).abs() < 1e-9);
    }

    #[test]
    fn test_theta_d_130te_anchor() {
        let effects = effects_with(IsotopeConfig {
            te_mass: Some(129.906),
            ..Default::default()
        });
        assert!(
            (effects.theta_d_substrate - 210.61).abs() < 0.05,
            "Θ_D anchor mismatch: got {}",
            effects.theta_d_substrate
        );
    }

    // --- Exotic-tier amplification and finiteness ---

    #[test]
    fn test_exotic_light_fe_amplifies_gap() {
        // α = 0.4 (default): lighter mass ⇒ larger gap; hypothetical 45 amu
        // Fe amplifies more than stable 54Fe, both above unity.
        let ratio = |fe_mass: f64| {
            let effects = effects_with(IsotopeConfig {
                fe_mass: Some(fe_mass),
                ..Default::default()
            });
            effects.delta_1_modified / DELTA_1_DEFAULT
        };
        let r_exotic = ratio(45.0);
        let r_stable = ratio(53.9396);
        assert!(
            r_exotic > r_stable && r_stable > 1.0,
            "expected exotic > stable > 1, got {} vs {}",
            r_exotic,
            r_stable
        );
    }

    #[test]
    fn test_exotic_extremes_finite() {
        // Range extremes of the exotic tier stay finite (exponent clamp).
        for (fe_mass, te_mass) in [(45.0, 105.0), (75.0, 145.0)] {
            let effects = effects_with(IsotopeConfig {
                fe_mass: Some(fe_mass),
                te_mass: Some(te_mass),
                ..Default::default()
            });
            assert!(effects.dw_factor_substrate.is_finite());
            assert!(effects.dw_factor_substrate > 0.0);
            assert!(effects.dw_factor_overlayer.is_finite());
            assert!(effects.delta_1_modified.is_finite());
            assert!(effects.delta_2_modified.is_finite());
            assert!(effects.coherence_length_modified.is_finite());
            assert!(effects.theta_d_substrate.is_finite());
            assert!(effects.theta_d_substrate > 0.0);
        }
    }
}
