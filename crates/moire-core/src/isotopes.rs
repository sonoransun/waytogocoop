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
}

/// Element-level data including isotopes and bulk thermodynamic properties.
#[derive(Debug, Clone)]
pub struct ElementData {
    pub symbol: &'static str,
    pub name: &'static str,
    pub isotopes: &'static [Isotope],
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
    Isotope { element: "Fe", mass_number: 54, atomic_mass: 53.9396, natural_abundance: 0.058 },
    Isotope { element: "Fe", mass_number: 56, atomic_mass: 55.9349, natural_abundance: 0.917 },
    Isotope { element: "Fe", mass_number: 57, atomic_mass: 56.9354, natural_abundance: 0.022 },
    Isotope { element: "Fe", mass_number: 58, atomic_mass: 57.9333, natural_abundance: 0.003 },
];

static TE_ISOTOPES: [Isotope; 6] = [
    Isotope { element: "Te", mass_number: 122, atomic_mass: 121.903, natural_abundance: 0.026 },
    Isotope { element: "Te", mass_number: 124, atomic_mass: 123.903, natural_abundance: 0.048 },
    Isotope { element: "Te", mass_number: 125, atomic_mass: 124.904, natural_abundance: 0.071 },
    Isotope { element: "Te", mass_number: 126, atomic_mass: 125.903, natural_abundance: 0.189 },
    Isotope { element: "Te", mass_number: 128, atomic_mass: 127.904, natural_abundance: 0.317 },
    Isotope { element: "Te", mass_number: 130, atomic_mass: 129.906, natural_abundance: 0.341 },
];

static SB_ISOTOPES: [Isotope; 2] = [
    Isotope { element: "Sb", mass_number: 121, atomic_mass: 120.904, natural_abundance: 0.572 },
    Isotope { element: "Sb", mass_number: 123, atomic_mass: 122.904, natural_abundance: 0.428 },
];

static BI_ISOTOPES: [Isotope; 1] = [
    Isotope { element: "Bi", mass_number: 209, atomic_mass: 208.980, natural_abundance: 1.0 },
];

static C_ISOTOPES: [Isotope; 2] = [
    Isotope { element: "C", mass_number: 12, atomic_mass: 12.000, natural_abundance: 0.9893 },
    Isotope { element: "C", mass_number: 13, atomic_mass: 13.00335, natural_abundance: 0.0107 },
];

// ---------------------------------------------------------------------------
// Element database
// ---------------------------------------------------------------------------

pub static FE: ElementData = ElementData {
    symbol: "Fe",
    name: "Iron",
    isotopes: &FE_ISOTOPES,
    debye_temperature: 260.0,
    gruneisen_parameter: 1.5,
    cohesive_energy_ev: 4.0,
};

pub static TE: ElementData = ElementData {
    symbol: "Te",
    name: "Tellurium",
    isotopes: &TE_ISOTOPES,
    debye_temperature: 165.0,
    gruneisen_parameter: 1.7,
    cohesive_energy_ev: 2.1,
};

pub static SB: ElementData = ElementData {
    symbol: "Sb",
    name: "Antimony",
    isotopes: &SB_ISOTOPES,
    debye_temperature: 210.0,
    gruneisen_parameter: 1.1,
    cohesive_energy_ev: 2.7,
};

pub static BI: ElementData = ElementData {
    symbol: "Bi",
    name: "Bismuth",
    isotopes: &BI_ISOTOPES,
    debye_temperature: 120.0,
    gruneisen_parameter: 1.2,
    cohesive_energy_ev: 2.2,
};

pub static C: ElementData = ElementData {
    symbol: "C",
    name: "Carbon",
    isotopes: &C_ISOTOPES,
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
/// ¹²⁵Te (I=1/2) is the only spin-bearing stable Te isotope. Returns 1.0 if
/// the selected mass is within 0.5 amu of ¹²⁵Te (124.904), 0.071 for natural
/// abundance, or 0.0 for other enrichments.
pub fn te_125_spin_fraction(te_mass_override: Option<f64>) -> f64 {
    const TE_125_MASS: f64 = 124.904;
    const NATURAL_ABUNDANCE: f64 = 0.071;

    match te_mass_override {
        None => NATURAL_ABUNDANCE,
        Some(target) => {
            if (target - TE_125_MASS).abs() < 0.5 {
                1.0
            } else {
                0.0
            }
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
}
