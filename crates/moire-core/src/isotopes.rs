//! Isotope data for elements in moire heterostructure materials.
//!
//! Provides atomic masses, natural abundances, and bulk thermodynamic
//! properties used by the speculative isotope-effects module.
//!
//! Data sources: AME2020 atomic mass evaluation, NUBASE2020 isotope tables.

/// A single isotope of an element.
#[derive(Debug, Clone, Copy)]
pub struct Isotope {
    pub element: &'static str,
    pub mass_number: u32,
    pub atomic_mass: f64,
    pub natural_abundance: f64,
    /// Half-life in seconds; `None` = stable.
    pub half_life_s: Option<f64>,
}

impl Isotope {
    const fn stable(
        element: &'static str,
        mass_number: u32,
        atomic_mass: f64,
        natural_abundance: f64,
    ) -> Self {
        Self { element, mass_number, atomic_mass, natural_abundance, half_life_s: None }
    }

    const fn synthetic(
        element: &'static str,
        mass_number: u32,
        atomic_mass: f64,
        half_life_s: f64,
    ) -> Self {
        Self {
            element,
            mass_number,
            atomic_mass,
            natural_abundance: 0.0,
            half_life_s: Some(half_life_s),
        }
    }
}

/// Element-level data including isotopes and bulk thermodynamic properties.
///
/// `isotopes` holds only the STABLE isotopes (natural averages and stable
/// slider ranges iterate it); radioactive species live in
/// `synthetic_isotopes`.
#[derive(Debug, Clone)]
pub struct ElementData {
    pub symbol: &'static str,
    pub name: &'static str,
    pub isotopes: &'static [Isotope],
    pub synthetic_isotopes: &'static [Isotope],
    pub debye_temperature: f64,
    pub gruneisen_parameter: f64,
    pub cohesive_energy_ev: f64,
}

/// User-configurable mass overrides for isotope enrichment.
///
/// `None` means "use natural abundance average".
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct IsotopeConfig {
    pub fe_mass: Option<f64>,
    pub te_mass: Option<f64>,
    pub sb_mass: Option<f64>,
    // Bi is monoisotopic — no override needed.
    #[serde(default)]
    pub c_mass: Option<f64>,
}

impl IsotopeConfig {
    /// Look up the effective mass for an element, falling back to natural average.
    pub fn effective_mass(&self, symbol: &str) -> f64 {
        match symbol {
            "Fe" => self.fe_mass.unwrap_or_else(|| natural_average_mass(&FE)),
            "Te" => self.te_mass.unwrap_or_else(|| natural_average_mass(&TE)),
            "Sb" => self.sb_mass.unwrap_or_else(|| natural_average_mass(&SB)),
            "Bi" => natural_average_mass(&BI),
            "C" => self.c_mass.unwrap_or_else(|| natural_average_mass(&C)),
            _ => 0.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Isotope tables
// ---------------------------------------------------------------------------

static FE_ISOTOPES: [Isotope; 4] = [
    Isotope::stable("Fe", 54, 53.9396, 0.058),
    Isotope::stable("Fe", 56, 55.9349, 0.917),
    Isotope::stable("Fe", 57, 56.9354, 0.022),
    Isotope::stable("Fe", 58, 57.9333, 0.003),
];

static TE_ISOTOPES: [Isotope; 6] = [
    Isotope::stable("Te", 122, 121.903, 0.026),
    Isotope::stable("Te", 124, 123.903, 0.048),
    Isotope::stable("Te", 125, 124.904, 0.071),
    Isotope::stable("Te", 126, 125.903, 0.189),
    Isotope::stable("Te", 128, 127.904, 0.317),
    Isotope::stable("Te", 130, 129.906, 0.341),
];

static SB_ISOTOPES: [Isotope; 2] = [
    Isotope::stable("Sb", 121, 120.904, 0.572),
    Isotope::stable("Sb", 123, 122.904, 0.428),
];

static BI_ISOTOPES: [Isotope; 1] = [Isotope::stable("Bi", 209, 208.980, 1.0)];

static C_ISOTOPES: [Isotope; 2] = [
    Isotope::stable("C", 12, 12.000, 0.9893),
    Isotope::stable("C", 13, 13.00335, 0.0107),
];

// ---------------------------------------------------------------------------
// Synthetic (radioactive) isotope tables — HIGHLY SPECULATIVE tier.
//
// Masses (amu) from AME2020; half-lives approximate. These never enter the
// natural averages or stable slider ranges above.
// ---------------------------------------------------------------------------

/// Half-life unit conversions to seconds.
const YEAR_S: f64 = 3.156e7;
const DAY_S: f64 = 86_400.0;
const HOUR_S: f64 = 3_600.0;
const MINUTE_S: f64 = 60.0;

static FE_SYNTHETIC: [Isotope; 4] = [
    Isotope::synthetic("Fe", 52, 51.9481, 8.28 * HOUR_S),
    Isotope::synthetic("Fe", 55, 54.9383, 2.74 * YEAR_S),
    Isotope::synthetic("Fe", 59, 58.9349, 44.5 * DAY_S),
    Isotope::synthetic("Fe", 60, 59.9341, 2.62e6 * YEAR_S),
];

static TE_SYNTHETIC: [Isotope; 4] = [
    Isotope::synthetic("Te", 121, 120.9049, 19.2 * DAY_S),
    Isotope::synthetic("Te", 127, 126.9052, 9.35 * HOUR_S),
    Isotope::synthetic("Te", 129, 128.9066, 69.6 * MINUTE_S),
    Isotope::synthetic("Te", 132, 131.9085, 3.20 * DAY_S),
];

static SB_SYNTHETIC: [Isotope; 3] = [
    Isotope::synthetic("Sb", 119, 118.9039, 38.2 * HOUR_S),
    Isotope::synthetic("Sb", 124, 123.9059, 60.2 * DAY_S),
    Isotope::synthetic("Sb", 125, 124.9053, 2.76 * YEAR_S),
];

static BI_SYNTHETIC: [Isotope; 3] = [
    Isotope::synthetic("Bi", 207, 206.9785, 31.6 * YEAR_S),
    Isotope::synthetic("Bi", 208, 207.9797, 3.68e5 * YEAR_S),
    Isotope::synthetic("Bi", 210, 209.9841, 5.01 * DAY_S),
];

static C_SYNTHETIC: [Isotope; 2] = [
    Isotope::synthetic("C", 11, 11.0114, 20.4 * MINUTE_S),
    Isotope::synthetic("C", 14, 14.0032, 5700.0 * YEAR_S),
];

// ---------------------------------------------------------------------------
// Hypothetical exploration mass ranges (amu) — HIGHLY SPECULATIVE.
//
// These extend far beyond the known isotopes toward the driplines; masses in
// between or beyond known isotopes are purely hypothetical what-if inputs.
// ---------------------------------------------------------------------------

pub const EXOTIC_MASS_RANGE_FE: (f64, f64) = (45.0, 75.0);
pub const EXOTIC_MASS_RANGE_TE: (f64, f64) = (105.0, 145.0);
pub const EXOTIC_MASS_RANGE_SB: (f64, f64) = (103.0, 140.0);
pub const EXOTIC_MASS_RANGE_C: (f64, f64) = (8.0, 22.0);

/// Hypothetical exploration range for an element's mass slider (amu).
pub fn exotic_mass_range(symbol: &str) -> Option<(f64, f64)> {
    match symbol {
        "Fe" => Some(EXOTIC_MASS_RANGE_FE),
        "Te" => Some(EXOTIC_MASS_RANGE_TE),
        "Sb" => Some(EXOTIC_MASS_RANGE_SB),
        "C" => Some(EXOTIC_MASS_RANGE_C),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Element database
// ---------------------------------------------------------------------------

pub static FE: ElementData = ElementData {
    symbol: "Fe",
    name: "Iron",
    isotopes: &FE_ISOTOPES,
    synthetic_isotopes: &FE_SYNTHETIC,
    debye_temperature: 260.0,
    gruneisen_parameter: 1.5,
    cohesive_energy_ev: 4.0,
};

pub static TE: ElementData = ElementData {
    symbol: "Te",
    name: "Tellurium",
    isotopes: &TE_ISOTOPES,
    synthetic_isotopes: &TE_SYNTHETIC,
    debye_temperature: 165.0,
    gruneisen_parameter: 1.7,
    cohesive_energy_ev: 2.1,
};

pub static SB: ElementData = ElementData {
    symbol: "Sb",
    name: "Antimony",
    isotopes: &SB_ISOTOPES,
    synthetic_isotopes: &SB_SYNTHETIC,
    debye_temperature: 210.0,
    gruneisen_parameter: 1.1,
    cohesive_energy_ev: 2.7,
};

pub static BI: ElementData = ElementData {
    symbol: "Bi",
    name: "Bismuth",
    isotopes: &BI_ISOTOPES,
    synthetic_isotopes: &BI_SYNTHETIC,
    debye_temperature: 120.0,
    gruneisen_parameter: 1.2,
    cohesive_energy_ev: 2.2,
};

pub static C: ElementData = ElementData {
    symbol: "C",
    name: "Carbon",
    isotopes: &C_ISOTOPES,
    synthetic_isotopes: &C_SYNTHETIC,
    debye_temperature: 2100.0, // graphene in-plane Debye temperature
    gruneisen_parameter: 1.8,
    cohesive_energy_ev: 7.4,
};

/// All elements in the database.
pub fn all_elements() -> [&'static ElementData; 5] {
    [&FE, &TE, &SB, &BI, &C]
}

/// Look up element data by symbol.
pub fn element_by_symbol(symbol: &str) -> Option<&'static ElementData> {
    match symbol {
        "Fe" => Some(&FE),
        "Te" => Some(&TE),
        "Sb" => Some(&SB),
        "Bi" => Some(&BI),
        "C" => Some(&C),
        _ => None,
    }
}

/// Abundance-weighted average atomic mass for an element (amu).
pub fn natural_average_mass(elem: &ElementData) -> f64 {
    elem.isotopes
        .iter()
        .map(|iso| iso.atomic_mass * iso.natural_abundance)
        .sum()
}

/// Estimate the ¹²⁵Te nuclear spin fraction for a given Te mass setting.
///
/// ¹²⁵Te (I=1/2) is the only spin-bearing stable Te isotope. `None` returns
/// the natural abundance (0.071). A mass override is interpreted as a binary
/// mixture of the two adjacent STABLE isotopes bracketing the target mass:
/// weight_lo = (m_hi − target)/(m_hi − m_lo); the ¹²⁵Te fraction is the
/// mixture weight belonging to the 124.904 endpoint if one of the brackets is
/// ¹²⁵Te, else 0.0. Below (above) the lightest (heaviest) stable isotope the
/// composition is taken as pure endpoint isotope.
///
/// NOTE: an override equal to the natural average mass (~126.62) is
/// interpreted as an enriched two-isotope blend (¹²⁶Te/¹²⁸Te), NOT natural
/// composition — pass `None` for natural.
pub fn te_125_spin_fraction(te_mass_override: Option<f64>) -> f64 {
    const TE_125_MASS: f64 = 124.904;
    const NATURAL_ABUNDANCE: f64 = 0.071;
    const MASS_EPS: f64 = 1e-6;

    let target = match te_mass_override {
        None => return NATURAL_ABUNDANCE,
        Some(t) => t,
    };

    let is_125 = |m: f64| (m - TE_125_MASS).abs() < MASS_EPS;

    // Stable isotope list is sorted by mass.
    let masses: Vec<f64> = TE.isotopes.iter().map(|i| i.atomic_mass).collect();
    let min_mass = masses[0];
    let max_mass = masses[masses.len() - 1];
    if target <= min_mass {
        return if is_125(min_mass) { 1.0 } else { 0.0 };
    }
    if target >= max_mass {
        return if is_125(max_mass) { 1.0 } else { 0.0 };
    }

    for pair in masses.windows(2) {
        let (m_lo, m_hi) = (pair[0], pair[1]);
        if target >= m_lo && target <= m_hi {
            let weight_lo = (m_hi - target) / (m_hi - m_lo);
            if is_125(m_lo) {
                return weight_lo;
            }
            if is_125(m_hi) {
                return 1.0 - weight_lo;
            }
            return 0.0;
        }
    }
    0.0
}

// ---------------------------------------------------------------------------
// Nearest-isotope classification (exotic tier helper)
// ---------------------------------------------------------------------------

/// Classification of a mass setting against the known isotope tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsotopeKind {
    Stable,
    Synthetic,
    Hypothetical,
}

/// Nearest known isotope to a requested mass, or "hypothetical" if none is
/// within 0.25 amu.
#[derive(Debug, Clone)]
pub struct NearestIsotope {
    /// e.g. "55Fe"; empty for hypothetical masses.
    pub label: String,
    pub kind: IsotopeKind,
    /// Half-life in seconds; `None` for stable isotopes (and hypothetical).
    pub half_life_s: Option<f64>,
}

/// Find the nearest known isotope (stable + synthetic) of `symbol` to
/// `mass_amu` and classify it. If no known isotope lies within 0.25 amu the
/// mass is classified as hypothetical.
pub fn nearest_isotope_info(symbol: &str, mass_amu: f64) -> NearestIsotope {
    const MATCH_WINDOW_AMU: f64 = 0.25;

    let hypothetical = NearestIsotope {
        label: String::new(),
        kind: IsotopeKind::Hypothetical,
        half_life_s: None,
    };

    let elem = match element_by_symbol(symbol) {
        Some(e) => e,
        None => return hypothetical,
    };

    let nearest = elem
        .isotopes
        .iter()
        .chain(elem.synthetic_isotopes.iter())
        .min_by(|a, b| {
            let da = (a.atomic_mass - mass_amu).abs();
            let db = (b.atomic_mass - mass_amu).abs();
            da.partial_cmp(&db).expect("isotope masses are finite")
        });

    match nearest {
        Some(iso) if (iso.atomic_mass - mass_amu).abs() <= MATCH_WINDOW_AMU => {
            let kind = if iso.half_life_s.is_none() {
                IsotopeKind::Stable
            } else {
                IsotopeKind::Synthetic
            };
            NearestIsotope {
                label: format!("{}{}", iso.mass_number, iso.element),
                kind,
                half_life_s: iso.half_life_s,
            }
        }
        _ => hypothetical,
    }
}

/// Humanize a half-life in seconds for display, e.g. "8.3 h", "2.7 y",
/// "2.6 My", "69.6 min". The threshold ladder mirrors the Python
/// `humanize_half_life` exactly so both apps render identical strings for
/// every database half-life (e.g. 69.6 min stays in minutes, 5700 y renders
/// as "5.7 ky").
pub fn humanize_half_life(seconds: f64) -> String {
    if seconds < 2.0 * HOUR_S {
        format!("{:.1} min", seconds / MINUTE_S)
    } else if seconds < 2.0 * DAY_S {
        format!("{:.1} h", seconds / HOUR_S)
    } else if seconds < 2.0 * YEAR_S {
        format!("{:.1} d", seconds / DAY_S)
    } else {
        let years = seconds / YEAR_S;
        if years < 1.0e3 {
            format!("{:.1} y", years)
        } else if years < 1.0e6 {
            format!("{:.1} ky", years / 1.0e3)
        } else {
            format!("{:.1} My", years / 1.0e6)
        }
    }
}

/// Material composition: element symbol and stoichiometric count.
pub fn material_composition(formula: &str) -> Option<&'static [(&'static str, u32)]> {
    match formula {
        "FeTe" => Some(&[("Fe", 1), ("Te", 1)]),
        "Sb2Te3" => Some(&[("Sb", 2), ("Te", 3)]),
        "Bi2Te3" => Some(&[("Bi", 2), ("Te", 3)]),
        "Sb2Te" => Some(&[("Sb", 2), ("Te", 1)]),
        // 2 carbon atoms per graphene hexagonal unit cell (A and B sublattices).
        "Graphene" => Some(&[("C", 2)]),
        // 4 / 6 carbon atoms per AB-bilayer / ABA-trilayer hexagonal unit cell.
        "Graphene-AB" => Some(&[("C", 4)]),
        "Graphene-ABA" => Some(&[("C", 6)]),
        _ => None,
    }
}

/// Average atomic mass per atom in a formula unit.
///
/// Uses the `IsotopeConfig` overrides when present, otherwise falls back
/// to the natural-abundance average.
pub fn formula_unit_avg_mass(formula: &str, config: &IsotopeConfig) -> Option<f64> {
    let comp = material_composition(formula)?;
    let mut total_mass = 0.0;
    let mut total_atoms = 0u32;
    for &(symbol, count) in comp {
        let m = config.effective_mass(symbol);
        total_mass += count as f64 * m;
        total_atoms += count;
    }
    if total_atoms == 0 {
        return None;
    }
    Some(total_mass / total_atoms as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abundances_sum_to_one() {
        for elem in all_elements() {
            let sum: f64 = elem.isotopes.iter().map(|i| i.natural_abundance).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "{} abundances sum to {}, expected ~1.0",
                elem.symbol,
                sum
            );
        }
    }

    #[test]
    fn test_natural_average_mass_fe() {
        let m = natural_average_mass(&FE);
        // IUPAC standard: 55.845
        assert!((m - 55.845).abs() < 0.1, "Fe avg mass = {}, expected ~55.845", m);
    }

    #[test]
    fn test_natural_average_mass_te() {
        let m = natural_average_mass(&TE);
        // IUPAC standard: 127.60 — our database omits minor isotopes (120Te, 123Te)
        // so the average is slightly lower (~126.6)
        assert!((m - 127.60).abs() < 1.1, "Te avg mass = {}, expected ~127.60", m);
    }

    #[test]
    fn test_formula_unit_avg_mass_fete() {
        let config = IsotopeConfig::default();
        let m = formula_unit_avg_mass("FeTe", &config).unwrap();
        let expected = (natural_average_mass(&FE) + natural_average_mass(&TE)) / 2.0;
        assert!((m - expected).abs() < 1e-10);
    }

    #[test]
    fn test_formula_unit_avg_mass_with_override() {
        let config = IsotopeConfig {
            fe_mass: Some(54.0),
            te_mass: None,
            sb_mass: None,
            c_mass: None,
        };
        let m = formula_unit_avg_mass("FeTe", &config).unwrap();
        let expected = (54.0 + natural_average_mass(&TE)) / 2.0;
        assert!((m - expected).abs() < 1e-10);
    }

    #[test]
    fn test_material_composition_known() {
        assert!(material_composition("FeTe").is_some());
        assert!(material_composition("Sb2Te3").is_some());
        assert!(material_composition("Bi2Te3").is_some());
        assert!(material_composition("Sb2Te").is_some());
        assert!(material_composition("Graphene-AB").is_some());
        assert!(material_composition("Graphene-ABA").is_some());
        assert!(material_composition("Unknown").is_none());
    }

    #[test]
    fn test_element_by_symbol() {
        assert_eq!(element_by_symbol("Fe").unwrap().symbol, "Fe");
        assert_eq!(element_by_symbol("Te").unwrap().symbol, "Te");
        assert_eq!(element_by_symbol("C").unwrap().symbol, "C");
        assert!(element_by_symbol("Zz").is_none());
    }

    #[test]
    fn test_natural_average_mass_carbon() {
        // IUPAC standard atomic weight: 12.011
        let m = natural_average_mass(&C);
        assert!((m - 12.011).abs() < 0.01, "C avg mass = {}, expected ~12.011", m);
    }

    #[test]
    fn test_graphene_composition() {
        // 2 carbon atoms per graphene hexagonal unit cell.
        let comp = material_composition("Graphene").expect("Graphene must be registered");
        assert_eq!(comp, &[("C", 2)]);
    }

    #[test]
    fn test_formula_unit_avg_mass_graphene() {
        let config = IsotopeConfig::default();
        let m = formula_unit_avg_mass("Graphene", &config).unwrap();
        // Average atomic mass per atom equals natural C mass (only one element).
        assert!((m - natural_average_mass(&C)).abs() < 1e-10);
    }

    #[test]
    fn test_graphene_c_mass_override() {
        let config = IsotopeConfig {
            c_mass: Some(13.00335), // pure C-13
            ..Default::default()
        };
        let m = formula_unit_avg_mass("Graphene", &config).unwrap();
        assert!((m - 13.00335).abs() < 1e-10);
    }

    #[test]
    fn test_stacked_graphene_composition() {
        assert_eq!(
            material_composition("Graphene-AB").unwrap(),
            &[("C", 4)]
        );
        assert_eq!(
            material_composition("Graphene-ABA").unwrap(),
            &[("C", 6)]
        );
    }

    #[test]
    fn test_formula_unit_avg_mass_stacked_graphene() {
        // Per-atom average of a single-element formula is the natural C mass.
        let config = IsotopeConfig::default();
        for formula in ["Graphene-AB", "Graphene-ABA"] {
            let m = formula_unit_avg_mass(formula, &config).unwrap();
            assert!((m - natural_average_mass(&C)).abs() < 1e-10, "{}", formula);
        }
    }

    #[test]
    fn test_stacked_graphene_c_mass_override() {
        let config = IsotopeConfig {
            c_mass: Some(13.00335), // pure C-13
            ..Default::default()
        };
        for formula in ["Graphene-AB", "Graphene-ABA"] {
            let m = formula_unit_avg_mass(formula, &config).unwrap();
            assert!((m - 13.00335).abs() < 1e-10, "{}", formula);
        }
    }

    // --- ¹²⁵Te spin fraction (two-isotope mixture model) ---

    #[test]
    fn test_spin_fraction_natural() {
        assert!((te_125_spin_fraction(None) - 0.071).abs() < 1e-12);
    }

    #[test]
    fn test_spin_fraction_pure_125() {
        assert!((te_125_spin_fraction(Some(124.904)) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_spin_fraction_midpoint_125_126() {
        // Halfway between 124.904 and 125.903 — 50/50 blend, 125Te is the low bracket.
        assert!((te_125_spin_fraction(Some(125.4035)) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_spin_fraction_midpoint_124_125() {
        // Halfway between 123.903 and 124.904 — 125Te is the high bracket.
        assert!((te_125_spin_fraction(Some(124.4035)) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_spin_fraction_far_from_125() {
        // Bracketed by 126Te/128Te — no 125Te content.
        assert!(te_125_spin_fraction(Some(127.0)).abs() < 1e-12);
    }

    #[test]
    fn test_spin_fraction_out_of_range() {
        // Below lightest stable (122Te) and above heaviest (130Te): pure
        // endpoint isotope, neither of which is 125Te.
        assert!(te_125_spin_fraction(Some(105.0)).abs() < 1e-12);
        assert!(te_125_spin_fraction(Some(145.0)).abs() < 1e-12);
    }

    // --- Synthetic isotope database ---

    #[test]
    fn test_synthetic_isotopes_have_half_life_and_zero_abundance() {
        for elem in all_elements() {
            assert!(
                !elem.synthetic_isotopes.is_empty(),
                "{} should have synthetic isotopes",
                elem.symbol
            );
            for iso in elem.synthetic_isotopes {
                let hl = iso.half_life_s.unwrap_or_else(|| {
                    panic!("synthetic {}{} missing half-life", iso.mass_number, iso.element)
                });
                assert!(hl > 0.0, "{}{} half-life must be > 0", iso.mass_number, iso.element);
                assert!(
                    iso.natural_abundance == 0.0,
                    "synthetic {}{} must have zero abundance",
                    iso.mass_number,
                    iso.element
                );
            }
        }
    }

    #[test]
    fn test_stable_isotopes_have_no_half_life() {
        for elem in all_elements() {
            for iso in elem.isotopes {
                assert!(iso.half_life_s.is_none());
            }
        }
    }

    // --- nearest_isotope_info ---

    #[test]
    fn test_nearest_isotope_synthetic_55fe() {
        let info = nearest_isotope_info("Fe", 54.94);
        assert_eq!(info.kind, IsotopeKind::Synthetic);
        assert_eq!(info.label, "55Fe");
        let hl = info.half_life_s.expect("55Fe must carry a half-life");
        assert!((hl - 2.74 * 3.156e7).abs() < 1.0);
    }

    #[test]
    fn test_nearest_isotope_hypothetical() {
        let info = nearest_isotope_info("Fe", 50.0);
        assert_eq!(info.kind, IsotopeKind::Hypothetical);
        assert!(info.label.is_empty());
        assert!(info.half_life_s.is_none());
    }

    #[test]
    fn test_nearest_isotope_stable_56fe() {
        let info = nearest_isotope_info("Fe", 55.9349);
        assert_eq!(info.kind, IsotopeKind::Stable);
        assert_eq!(info.label, "56Fe");
        assert!(info.half_life_s.is_none());
    }

    #[test]
    fn test_humanize_half_life() {
        assert_eq!(humanize_half_life(8.28 * 3_600.0), "8.3 h");
        assert_eq!(humanize_half_life(2.74 * 3.156e7), "2.7 y");
        assert_eq!(humanize_half_life(2.62e6 * 3.156e7), "2.6 My");
        assert_eq!(humanize_half_life(69.6 * 60.0), "69.6 min");
    }

    #[test]
    fn test_exotic_mass_ranges() {
        assert_eq!(exotic_mass_range("Fe"), Some((45.0, 75.0)));
        assert_eq!(exotic_mass_range("Te"), Some((105.0, 145.0)));
        assert_eq!(exotic_mass_range("Sb"), Some((103.0, 140.0)));
        assert_eq!(exotic_mass_range("C"), Some((8.0, 22.0)));
        assert_eq!(exotic_mass_range("Bi"), None);
    }
}
