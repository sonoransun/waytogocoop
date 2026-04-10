/// Lattice symmetry classification.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LatticeType {
    Hexagonal,
    Square,
}

/// A crystallographic material used in moire superlattice calculations.
#[derive(Debug, Clone)]
pub struct Material {
    pub name: &'static str,
    pub formula: &'static str,
    pub lattice_type: LatticeType,
    /// In-plane lattice constant in Angstroms.
    pub a: f64,
    /// Out-of-plane lattice constant in Angstroms.
    pub c: f64,
    pub space_group: &'static str,
    /// Role: "substrate" or "overlayer".
    pub role: &'static str,
}

static MATERIALS: &[Material] = &[
    Material {
        name: "FeTe",
        formula: "FeTe",
        lattice_type: LatticeType::Square,
        a: 3.82,
        c: 6.25,
        space_group: "P4/nmm",
        role: "substrate",
    },
    Material {
        name: "Sb2Te3",
        formula: "Sb₂Te₃",
        lattice_type: LatticeType::Hexagonal,
        a: 4.264,
        c: 30.458,
        space_group: "R-3m",
        role: "overlayer",
    },
    Material {
        name: "Bi2Te3",
        formula: "Bi₂Te₃",
        lattice_type: LatticeType::Hexagonal,
        a: 4.386,
        c: 30.497,
        space_group: "R-3m",
        role: "overlayer",
    },
    Material {
        name: "Sb2Te",
        formula: "Sb₂Te",
        lattice_type: LatticeType::Hexagonal,
        a: 4.272,
        c: 17.633,
        space_group: "R-3m",
        role: "overlayer",
    },
    Material {
        name: "Graphene",
        formula: "Graphene",
        lattice_type: LatticeType::Hexagonal,
        a: 2.46,
        c: 3.35,
        space_group: "P6/mmm",
        // Role "both": graphene can act as substrate or overlayer for
        // twisted-bilayer homo-bilayer moire systems.
        // Honeycomb basis is approximated as first-order hexagonal Bravais;
        // AA/AB stacking and flat-band physics are NOT resolved.
        role: "both",
    },
];

/// Returns all materials in the database.
pub fn all_materials() -> &'static [Material] {
    MATERIALS
}

/// Returns all materials usable as a substrate (role "substrate" or "both").
pub fn substrates() -> Vec<&'static Material> {
    MATERIALS
        .iter()
        .filter(|m| m.role == "substrate" || m.role == "both")
        .collect()
}

/// Returns the default substrate material (first entry with substrate role).
///
/// Retained for backwards compatibility; prefer `substrates()` for selection.
pub fn substrate() -> &'static Material {
    substrates()
        .into_iter()
        .next()
        .expect("at least one substrate must be defined")
}

/// Returns all materials usable as an overlayer (role "overlayer" or "both").
pub fn overlayers() -> Vec<&'static Material> {
    MATERIALS
        .iter()
        .filter(|m| m.role == "overlayer" || m.role == "both")
        .collect()
}

/// Look up a material by name (case-sensitive).
pub fn by_name(name: &str) -> Option<&'static Material> {
    MATERIALS.iter().find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_materials_count() {
        assert_eq!(all_materials().len(), 5);
    }

    #[test]
    fn test_substrate_is_fete() {
        let sub = substrate();
        assert_eq!(sub.name, "FeTe");
        assert_eq!(sub.lattice_type, LatticeType::Square);
        assert!((sub.a - 3.82).abs() < 1e-10);
    }

    #[test]
    fn test_substrates_includes_graphene() {
        let subs = substrates();
        // FeTe (substrate) + Graphene (both) = 2
        assert_eq!(subs.len(), 2);
        assert!(subs.iter().any(|m| m.name == "FeTe"));
        assert!(subs.iter().any(|m| m.name == "Graphene"));
    }

    #[test]
    fn test_overlayers_count() {
        // 3 TI overlayers + Graphene (both) = 4
        let overs = overlayers();
        assert_eq!(overs.len(), 4);
        for m in &overs {
            assert!(m.role == "overlayer" || m.role == "both");
        }
    }

    #[test]
    fn test_graphene_role_both() {
        let g = by_name("Graphene").unwrap();
        assert_eq!(g.role, "both");
        assert_eq!(g.lattice_type, LatticeType::Hexagonal);
        assert!((g.a - 2.46).abs() < 1e-10);
        // Should appear in both filtered lists.
        assert!(substrates().iter().any(|m| m.name == "Graphene"));
        assert!(overlayers().iter().any(|m| m.name == "Graphene"));
    }

    #[test]
    fn test_by_name_found() {
        let m = by_name("Bi2Te3").unwrap();
        assert_eq!(m.formula, "Bi₂Te₃");
        assert_eq!(m.lattice_type, LatticeType::Hexagonal);
    }

    #[test]
    fn test_by_name_graphene() {
        let m = by_name("Graphene").unwrap();
        assert_eq!(m.formula, "Graphene");
        assert_eq!(m.space_group, "P6/mmm");
    }

    #[test]
    fn test_by_name_not_found() {
        assert!(by_name("Unobtanium").is_none());
    }
}
