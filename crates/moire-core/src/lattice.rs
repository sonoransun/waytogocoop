use nalgebra::Vector2;
use std::f64::consts::PI;

/// A 2D Bravais lattice defined by two basis vectors.
#[derive(Debug, Clone)]
pub struct Lattice2D {
    pub a1: Vector2<f64>,
    pub a2: Vector2<f64>,
}

impl Lattice2D {
    /// Construct a hexagonal lattice with parameter `a`.
    /// a1 = (a, 0), a2 = (a/2, a*sqrt(3)/2).
    pub fn hexagonal(a: f64) -> Self {
        Self {
            a1: Vector2::new(a, 0.0),
            a2: Vector2::new(a / 2.0, a * 3.0_f64.sqrt() / 2.0),
        }
    }

    /// Construct a square lattice with parameter `a`.
    /// a1 = (a, 0), a2 = (0, a).
    pub fn square(a: f64) -> Self {
        Self {
            a1: Vector2::new(a, 0.0),
            a2: Vector2::new(0.0, a),
        }
    }

    /// Return a new lattice rotated by `angle_rad` radians.
    pub fn rotated(&self, angle_rad: f64) -> Self {
        let c = angle_rad.cos();
        let s = angle_rad.sin();
        let rotate = |v: &Vector2<f64>| Vector2::new(v.x * c - v.y * s, v.x * s + v.y * c);
        Self {
            a1: rotate(&self.a1),
            a2: rotate(&self.a2),
        }
    }

    /// Compute the reciprocal lattice vectors.
    /// b1 = 2*pi * (a2_perp) / (a1 . a2_perp)
    /// b2 = 2*pi * (a1_perp) / (a2 . a1_perp)
    /// where v_perp = (-v.y, v.x) for 90-degree CCW rotation in 2D.
    pub fn reciprocal(&self) -> Result<Self, String> {
        let a2_perp = Vector2::new(-self.a2.y, self.a2.x);
        let a1_perp = Vector2::new(-self.a1.y, self.a1.x);
        let denom1 = self.a1.dot(&a2_perp);
        let denom2 = self.a2.dot(&a1_perp);
        if denom1.abs() < 1e-15 || denom2.abs() < 1e-15 {
            return Err("Degenerate lattice: basis vectors are collinear".to_string());
        }
        Ok(Self {
            a1: 2.0 * PI * a2_perp / denom1,
            a2: 2.0 * PI * a1_perp / denom2,
        })
    }

    /// Generate lattice points for indices in -nx..=nx, -ny..=ny.
    pub fn generate_points(&self, nx: i32, ny: i32) -> Vec<Vector2<f64>> {
        let mut points = Vec::with_capacity(((2 * nx + 1) * (2 * ny + 1)) as usize);
        for i in -nx..=nx {
            for j in -ny..=ny {
                points.push(self.a1 * i as f64 + self.a2 * j as f64);
            }
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square_lattice() {
        let lat = Lattice2D::square(3.82);
        assert!((lat.a1.x - 3.82).abs() < 1e-10);
        assert!(lat.a1.y.abs() < 1e-10);
        assert!(lat.a2.x.abs() < 1e-10);
        assert!((lat.a2.y - 3.82).abs() < 1e-10);
    }

    #[test]
    fn test_hexagonal_lattice() {
        let lat = Lattice2D::hexagonal(4.264);
        assert!((lat.a1.x - 4.264).abs() < 1e-10);
        assert!(lat.a1.y.abs() < 1e-10);
        assert!((lat.a2.x - 4.264 / 2.0).abs() < 1e-10);
        assert!((lat.a2.y - 4.264 * 3.0_f64.sqrt() / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_identity() {
        let lat = Lattice2D::square(1.0);
        let rot = lat.rotated(0.0);
        assert!((rot.a1 - lat.a1).norm() < 1e-10);
        assert!((rot.a2 - lat.a2).norm() < 1e-10);
    }

    #[test]
    fn test_rotation_90_deg() {
        let lat = Lattice2D::square(1.0);
        let rot = lat.rotated(PI / 2.0);
        // a1 = (1,0) -> (0,1)
        assert!((rot.a1.x).abs() < 1e-10);
        assert!((rot.a1.y - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocal_square() {
        let lat = Lattice2D::square(1.0);
        let rec = lat.reciprocal().unwrap();
        // For a square lattice with a=1: b1 = (2pi, 0), b2 = (0, 2pi)
        assert!((rec.a1.x - 2.0 * PI).abs() < 1e-10);
        assert!(rec.a1.y.abs() < 1e-10);
        assert!(rec.a2.x.abs() < 1e-10);
        assert!((rec.a2.y - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocal_orthogonality() {
        let lat = Lattice2D::hexagonal(4.264);
        let rec = lat.reciprocal().unwrap();
        // a1 . b2 = 0, a2 . b1 = 0
        assert!((lat.a1.dot(&rec.a2)).abs() < 1e-10);
        assert!((lat.a2.dot(&rec.a1)).abs() < 1e-10);
        // a1 . b1 = 2pi, a2 . b2 = 2pi
        assert!((lat.a1.dot(&rec.a1) - 2.0 * PI).abs() < 1e-10);
        assert!((lat.a2.dot(&rec.a2) - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_generate_points_count() {
        let lat = Lattice2D::square(1.0);
        let pts = lat.generate_points(2, 3);
        assert_eq!(pts.len(), 5 * 7); // (2*2+1) * (2*3+1)
    }

    #[test]
    fn test_generate_points_origin() {
        let lat = Lattice2D::square(1.0);
        let pts = lat.generate_points(1, 1);
        // The origin (0,0) should be among the points
        assert!(pts.iter().any(|p| p.norm() < 1e-10));
    }

    #[test]
    fn test_reciprocal_degenerate_err() {
        let lat = Lattice2D {
            a1: Vector2::new(1.0, 0.0),
            a2: Vector2::new(2.0, 0.0), // collinear with a1
        };
        assert!(lat.reciprocal().is_err());
    }

    #[test]
    fn test_reciprocal_zero_vectors_err() {
        let lat = Lattice2D {
            a1: Vector2::new(0.0, 0.0),
            a2: Vector2::new(0.0, 0.0),
        };
        assert!(lat.reciprocal().is_err());
    }
}
