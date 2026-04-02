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
];

/// Returns all materials in the database.
pub fn all_materials() -> &'static [Material] {
    MATERIALS
}

/// Returns the substrate material (FeTe).
pub fn substrate() -> &'static Material {
    &MATERIALS[0]
}

/// Returns all overlayer materials.
pub fn overlayers() -> &'static [Material] {
    &MATERIALS[1..]
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
        assert_eq!(all_materials().len(), 4);
    }

    #[test]
    fn test_substrate_is_fete() {
        let sub = substrate();
        assert_eq!(sub.name, "FeTe");
        assert_eq!(sub.lattice_type, LatticeType::Square);
        assert!((sub.a - 3.82).abs() < 1e-10);
    }

    #[test]
    fn test_overlayers_count() {
        assert_eq!(overlayers().len(), 3);
        for m in overlayers() {
            assert_eq!(m.role, "overlayer");
        }
    }

    #[test]
    fn test_by_name_found() {
        let m = by_name("Bi2Te3").unwrap();
        assert_eq!(m.formula, "Bi₂Te₃");
        assert_eq!(m.lattice_type, LatticeType::Hexagonal);
    }

    #[test]
    fn test_by_name_not_found() {
        assert!(by_name("Unobtanium").is_none());
    }
}
