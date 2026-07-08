//! Twisted and stacked graphene: moire patterns and magic-angle flat-band
//! superconductivity.
//!
//! Established physics unless marked **SPECULATIVE**.
//!
//! Literature basis for the Bistritzer-MacDonald continuum model and
//! flat-band superconductivity:
//!   - Bistritzer & MacDonald, PNAS 108, 12233 (2011) — magic-angle condition
//!   - Cao et al., Nature 556, 43 (2018) — superconductivity in magic-angle TBG
//!   - Park et al., Nature 590, 249 (2021) — alternating-twist trilayer graphene
//!   - Khalaf et al., PRB 100, 085109 (2019) — sqrt(2) magic-angle enhancement
//!     in alternating-twist multilayers
//!
//! Mirrors `src/waytogocoop/computation/graphene.py` in the Python stack —
//! changes here must land in both.

use crate::materials::LatticeType;
use crate::moire::{moire_periodicity_1d, moire_periodicity_twist};
use nalgebra::Vector2;
use rayon::prelude::*;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Physical constants
// ---------------------------------------------------------------------------

/// Graphene in-plane lattice constant in Angstrom.
pub const GRAPHENE_A: f64 = 2.46;
/// Dirac velocity as hbar*v_F in eV*Angstrom.
const HBAR_VF_GRAPHENE: f64 = 5.96;
/// Interlayer tunnelling amplitude w in eV (Bistritzer-MacDonald).
const W_INTERLAYER_TBG: f64 = 0.110;
/// Boltzmann constant in eV/K.
const KB_EV_K: f64 = 8.617333262e-5;
/// Weak-coupling BCS ratio Delta / (k_B * Tc).
const BCS_GAP_RATIO: f64 = 1.764;
/// Peak superconducting gap for magic-angle bilayer in meV.
const DELTA_TBG_MAX: f64 = 0.30;
/// Peak superconducting gap for alternating-twist trilayer in meV.
const DELTA_TTG_MAX: f64 = 0.44;
/// Lorentzian width of the superconducting twist-angle window in degrees.
const THETA_SC_WIDTH_DEG: f64 = 0.1;
/// Optimal moire-band filling nu for superconductivity.
const NU_OPTIMAL_FILLING: f64 = 2.4;
/// Half-width of the superconducting filling dome.
const NU_DOME_WIDTH: f64 = 0.8;
/// Ginzburg-Landau coherence length of magic-angle TBG in Angstrom.
pub const XI_TBG: f64 = 500.0;
/// Stacking-registry phase coefficients for the angle-ordered hexagonal
/// G shell (0..300 deg in 60-deg steps): an AB offset tau = (a1 + a2)/3
/// shifts each plane wave by G_n . tau = c_n * 2*pi/3 with these c_n.
const HEX_PHASE_COEFFS: [f64; 6] = [1.0, 1.0, 0.0, -1.0, -1.0, 0.0];
/// Poisson ratio of graphene (in-plane, Blakslee 1970).
pub const POISSON_GRAPHENE: f64 = 0.16;

// ---------------------------------------------------------------------------
// Stacking registry
// ---------------------------------------------------------------------------

/// Stacking arrangement of a graphene multilayer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum StackingKind {
    AA,
    AB,
    #[default]
    TwistedBilayer,
    ABA,
    ABC,
    AlternatingTrilayer,
}

impl StackingKind {
    pub const ALL: [StackingKind; 6] = [
        StackingKind::AA,
        StackingKind::AB,
        StackingKind::TwistedBilayer,
        StackingKind::ABA,
        StackingKind::ABC,
        StackingKind::AlternatingTrilayer,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            StackingKind::AA => "AA bilayer",
            StackingKind::AB => "AB (Bernal) bilayer",
            StackingKind::TwistedBilayer => "Twisted bilayer",
            StackingKind::ABA => "ABA trilayer",
            StackingKind::ABC => "ABC trilayer",
            StackingKind::AlternatingTrilayer => "Alt-twist trilayer",
        }
    }

    pub fn n_layers(&self) -> usize {
        match self {
            StackingKind::AA | StackingKind::AB | StackingKind::TwistedBilayer => 2,
            StackingKind::ABA | StackingKind::ABC | StackingKind::AlternatingTrilayer => 3,
        }
    }

    pub fn is_twisted(&self) -> bool {
        matches!(
            self,
            StackingKind::TwistedBilayer | StackingKind::AlternatingTrilayer
        )
    }
}

/// One graphene layer within a stack: rigid twist plus stacking registry.
#[derive(Debug, Clone, Copy)]
pub struct LayerSpec {
    /// Twist angle of the layer in degrees.
    pub twist_deg: f64,
    /// Registry index: 0 = A, 1 = B, 2 = C stacking positions.
    pub stacking_index: i32,
}

/// Expand a stacking kind into per-layer twist angles and registry indices.
pub fn stack_layers(kind: StackingKind, twist_angle_deg: f64) -> Vec<LayerSpec> {
    let layer = |twist_deg: f64, stacking_index: i32| LayerSpec {
        twist_deg,
        stacking_index,
    };
    match kind {
        StackingKind::AA => vec![layer(0.0, 0), layer(0.0, 0)],
        StackingKind::AB => vec![layer(0.0, 0), layer(0.0, 1)],
        StackingKind::TwistedBilayer => vec![layer(0.0, 0), layer(twist_angle_deg, 0)],
        StackingKind::ABA => vec![layer(0.0, 0), layer(0.0, 1), layer(0.0, 0)],
        StackingKind::ABC => vec![layer(0.0, 0), layer(0.0, 1), layer(0.0, 2)],
        StackingKind::AlternatingTrilayer => {
            vec![layer(0.0, 0), layer(twist_angle_deg, 0), layer(0.0, 0)]
        }
    }
}

// ---------------------------------------------------------------------------
// Stack moire pattern
// ---------------------------------------------------------------------------

/// Configuration for a graphene stack moire pattern computation.
#[derive(Debug, Clone)]
pub struct GrapheneStackConfig {
    pub stacking: StackingKind,
    /// Twist angle in degrees (applies to twisted stackings only).
    pub twist_angle_deg: f64,
    /// In-plane lattice constant in Angstrom.
    pub lattice_a: f64,
    /// Grid resolution (NxN pixels).
    pub resolution: usize,
    /// Viewport size in Angstroms.
    pub physical_extent: f64,
}

impl Default for GrapheneStackConfig {
    fn default() -> Self {
        Self {
            stacking: StackingKind::TwistedBilayer,
            twist_angle_deg: 1.08,
            lattice_a: GRAPHENE_A,
            resolution: 256,
            physical_extent: 400.0,
        }
    }
}

/// Result of a graphene stack moire computation.
#[derive(Debug, Clone)]
pub struct GrapheneStackResult {
    /// Row-major intensity values, resolution x resolution, normalized to [0, 1].
    pub pattern: Vec<f64>,
    pub resolution: usize,
    pub physical_extent: f64,
    /// Estimated moire period in Angstroms (infinite for untwisted stacks).
    pub moire_period: f64,
    pub n_layers: usize,
}

/// First hexagonal reciprocal shell, angle-ordered 0..300 deg,
/// magnitude 4*pi / (a*sqrt(3)).
pub fn hex_g_vectors(a: f64) -> [Vector2<f64>; 6] {
    let mag = 4.0 * PI / (a * 3.0_f64.sqrt());
    std::array::from_fn(|i| {
        let angle = i as f64 * PI / 3.0;
        Vector2::new(mag * angle.cos(), mag * angle.sin())
    })
}

/// Rotate a hexagonal G shell rigidly by `angle_rad`.
fn rotate_hex_g_vectors(gs: &[Vector2<f64>; 6], angle_rad: f64) -> [Vector2<f64>; 6] {
    let c = angle_rad.cos();
    let s = angle_rad.sin();
    std::array::from_fn(|i| Vector2::new(gs[i].x * c - gs[i].y * s, gs[i].x * s + gs[i].y * c))
}

/// Apply uniaxial heterostrain to a hexagonal G shell: G' = (I - E) G with
/// E = R(phi) diag(eps, -poisson*eps) R(phi)^T, eps = strain_percent / 100,
/// phi = strain_angle_deg (tension axis in-plane).
pub fn strained_g_vectors(
    gs: &[Vector2<f64>; 6],
    strain_percent: f64,
    strain_angle_deg: f64,
    poisson: f64,
) -> [Vector2<f64>; 6] {
    let eps = strain_percent / 100.0;
    let phi = strain_angle_deg.to_radians();
    let (c, s) = (phi.cos(), phi.sin());
    let (e1, e2) = (eps, -poisson * eps);
    let e_xx = e1 * c * c + e2 * s * s;
    let e_yy = e1 * s * s + e2 * c * c;
    let e_xy = (e1 - e2) * c * s;
    std::array::from_fn(|i| {
        let g = gs[i];
        Vector2::new(
            g.x - (e_xx * g.x + e_xy * g.y),
            g.y - (e_xy * g.x + e_yy * g.y),
        )
    })
}

/// Moire period from two first-shell hexagonal G sets (index-matched):
/// L = 4*pi / (sqrt(3) * min_n |G_a[n] - G_b[n]|), infinite for degenerate
/// shells.  Handles twist, heterostrain and lattice mismatch uniformly.
pub fn moire_period_from_g_shells(g_a: &[Vector2<f64>; 6], g_b: &[Vector2<f64>; 6]) -> f64 {
    let min_dg = g_a
        .iter()
        .zip(g_b.iter())
        .map(|(a, b)| (a - b).norm())
        .fold(f64::MAX, f64::min);
    if min_dg < 1e-12 {
        return f64::INFINITY;
    }
    4.0 * PI / (3.0_f64.sqrt() * min_dg)
}

/// Potential of one layer at point r given its rotated G shell and registry
/// phase phi: V(r) = sum_n [cos(theta_n) (+ cos(theta_n - c_n*2*pi/3) if
/// honeycomb)], theta_n = G_n . r - c_n * phi.  The honeycomb term adds the
/// B sublattice at tau_B = (a1 + a2)/3.
#[inline]
fn layer_potential_from_gs(
    gs: &[Vector2<f64>; 6],
    phi: f64,
    r: Vector2<f64>,
    honeycomb: bool,
) -> f64 {
    gs.iter()
        .zip(HEX_PHASE_COEFFS.iter())
        .map(|(g, c)| {
            let theta = g.dot(&r) - c * phi;
            if honeycomb {
                theta.cos() + (theta - c * 2.0 * PI / 3.0).cos()
            } else {
                theta.cos()
            }
        })
        .sum()
}

/// Potential of a single graphene layer at (x, y): rigid twist, stacking
/// registry (0 = A, 1 = B, 2 = C) and optional B-sublattice (honeycomb) term.
pub fn layer_potential(
    x: f64,
    y: f64,
    lattice_a: f64,
    twist_deg: f64,
    stacking_index: i32,
    honeycomb: bool,
) -> f64 {
    let gs = rotate_hex_g_vectors(&hex_g_vectors(lattice_a), twist_deg.to_radians());
    let phi = stacking_index as f64 * 2.0 * PI / 3.0;
    layer_potential_from_gs(&gs, phi, Vector2::new(x, y), honeycomb)
}

/// Validate the shared grid / lattice parameters of a stack config.
fn validate_stack_config(cfg: &GrapheneStackConfig) -> Result<(), String> {
    if cfg.lattice_a <= 0.0 {
        return Err(format!("lattice_a must be positive, got {}", cfg.lattice_a));
    }
    if cfg.resolution < 2 {
        return Err(format!("resolution must be >= 2, got {}", cfg.resolution));
    }
    if cfg.physical_extent <= 0.0 {
        return Err(format!(
            "physical_extent must be positive, got {}",
            cfg.physical_extent
        ));
    }
    Ok(())
}

/// Raw (unnormalized) product-of-layers potential on the NxN grid, row-major.
/// `displacement` warps the sampling point r -> r + u(r) for all layers;
/// `overlayer_gs` multiplies in an extra substrate potential sum cos(G . r).
fn compute_stack_raw(
    layer_data: &[([Vector2<f64>; 6], f64)],
    n: usize,
    extent: f64,
    honeycomb: bool,
    displacement: Option<(&[f64], &[f64])>,
    overlayer_gs: Option<&[Vector2<f64>]>,
) -> Vec<f64> {
    (0..n)
        .into_par_iter()
        .flat_map_iter(|iy| {
            let y = (iy as f64 / (n - 1).max(1) as f64 - 0.5) * extent;
            (0..n).map(move |ix| {
                let x = (ix as f64 / (n - 1).max(1) as f64 - 0.5) * extent;
                let r = match displacement {
                    Some((u_x, u_y)) => {
                        let idx = iy * n + ix;
                        Vector2::new(x + u_x[idx], y + u_y[idx])
                    }
                    None => Vector2::new(x, y),
                };
                let mut val = 1.0_f64;
                for (gs, phi) in layer_data {
                    val *= layer_potential_from_gs(gs, *phi, r, honeycomb);
                }
                if let Some(over) = overlayer_gs {
                    val *= over.iter().map(|g| g.dot(&r).cos()).sum::<f64>();
                }
                val
            })
        })
        .collect()
}

/// Normalize a pattern to [0, 1] in place (constant fields collapse to 0.5).
fn normalize_unit(pattern: &mut [f64]) {
    let min_val = pattern.iter().cloned().fold(f64::MAX, f64::min);
    let max_val = pattern.iter().cloned().fold(f64::MIN, f64::max);
    let range = max_val - min_val;
    if range > 1e-15 {
        for v in pattern.iter_mut() {
            *v = (*v - min_val) / range;
        }
    } else {
        for v in pattern.iter_mut() {
            *v = 0.5;
        }
    }
}

/// Rotated G shell and registry phase for every layer of a stack.
fn stack_layer_data(
    layers: &[LayerSpec],
    base_gs: &[Vector2<f64>; 6],
) -> Vec<([Vector2<f64>; 6], f64)> {
    layers
        .iter()
        .map(|layer| {
            (
                rotate_hex_g_vectors(base_gs, layer.twist_deg.to_radians()),
                layer.stacking_index as f64 * 2.0 * PI / 3.0,
            )
        })
        .collect()
}

/// Compute the multilayer graphene moire pattern via plane-wave superposition.
///
/// Each layer contributes V_i(r) = sum_n cos(G_n . r - c_n * phi_i) with the
/// registry phase phi_i = stacking_index * 2*pi/3; the pattern is the product
/// over layers, normalized to [0, 1].
pub fn compute_graphene_stack(cfg: &GrapheneStackConfig) -> Result<GrapheneStackResult, String> {
    validate_stack_config(cfg)?;

    let n = cfg.resolution;
    let extent = cfg.physical_extent;
    let layers = stack_layers(cfg.stacking, cfg.twist_angle_deg);
    let layer_data = stack_layer_data(&layers, &hex_g_vectors(cfg.lattice_a));

    let mut pattern = compute_stack_raw(&layer_data, n, extent, false, None, None);
    normalize_unit(&mut pattern);

    let twists: Vec<f64> = layers.iter().map(|l| l.twist_deg).collect();
    let theta_rel = twists.iter().cloned().fold(f64::MIN, f64::max)
        - twists.iter().cloned().fold(f64::MAX, f64::min);
    let moire_period = if theta_rel.abs() > 1e-6 {
        moire_periodicity_twist(cfg.lattice_a, theta_rel)
    } else {
        f64::INFINITY
    };

    Ok(GrapheneStackResult {
        pattern,
        resolution: n,
        physical_extent: extent,
        moire_period,
        n_layers: layers.len(),
    })
}

// ---------------------------------------------------------------------------
// Stack moire pattern v2: honeycomb basis, heterostrain, displacement warp
// ---------------------------------------------------------------------------

/// Extended stack configuration: honeycomb (two-sublattice) basis and
/// uniaxial heterostrain applied to layer index 1 before its twist.
#[derive(Debug, Clone)]
pub struct GrapheneStackConfigV2 {
    pub base: GrapheneStackConfig,
    /// Include the B-sublattice plane-wave term of each layer.
    pub honeycomb: bool,
    /// Uniaxial heterostrain on layer index 1 in percent.
    pub heterostrain_percent: f64,
    /// In-plane tension axis of the heterostrain in degrees.
    pub heterostrain_angle_deg: f64,
}

impl Default for GrapheneStackConfigV2 {
    fn default() -> Self {
        Self {
            base: GrapheneStackConfig::default(),
            honeycomb: false,
            heterostrain_percent: 0.0,
            heterostrain_angle_deg: 0.0,
        }
    }
}

/// Per-layer G shells and registry phases for a v2 stack: heterostrain is
/// applied to layer index 1 pre-twist (strain the aligned shell, then rotate).
fn stack_layer_data_v2(cfg: &GrapheneStackConfigV2) -> Vec<([Vector2<f64>; 6], f64)> {
    let layers = stack_layers(cfg.base.stacking, cfg.base.twist_angle_deg);
    let base_gs = hex_g_vectors(cfg.base.lattice_a);
    layers
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let shell = if i == 1 && cfg.heterostrain_percent != 0.0 {
                strained_g_vectors(
                    &base_gs,
                    cfg.heterostrain_percent,
                    cfg.heterostrain_angle_deg,
                    POISSON_GRAPHENE,
                )
            } else {
                base_gs
            };
            (
                rotate_hex_g_vectors(&shell, layer.twist_deg.to_radians()),
                layer.stacking_index as f64 * 2.0 * PI / 3.0,
            )
        })
        .collect()
}

fn validate_stack_config_v2(
    cfg: &GrapheneStackConfigV2,
    displacement: Option<(&[f64], &[f64])>,
) -> Result<(), String> {
    validate_stack_config(&cfg.base)?;
    if cfg.heterostrain_percent.abs() >= 100.0 {
        return Err(format!(
            "heterostrain_percent must satisfy |eps| < 100, got {}",
            cfg.heterostrain_percent
        ));
    }
    if let Some((u_x, u_y)) = displacement {
        let expected = cfg.base.resolution * cfg.base.resolution;
        if u_x.len() != expected || u_y.len() != expected {
            return Err(format!(
                "displacement fields must have len resolution^2 = {}, got ({}, {})",
                expected,
                u_x.len(),
                u_y.len()
            ));
        }
    }
    Ok(())
}

/// Compute the multilayer graphene moire pattern with optional honeycomb
/// basis, heterostrain on layer index 1 and an in-plane displacement warp
/// r -> r + u(r) applied to all layers (`displacement` = (u_x, u_y), each a
/// row-major resolution^2 array in Angstrom).  With default flags and no
/// displacement this reproduces `compute_graphene_stack` bit-for-bit.
pub fn compute_graphene_stack_v2(
    cfg: &GrapheneStackConfigV2,
    displacement: Option<(&[f64], &[f64])>,
) -> Result<GrapheneStackResult, String> {
    validate_stack_config_v2(cfg, displacement)?;

    let n = cfg.base.resolution;
    let extent = cfg.base.physical_extent;
    let layer_data = stack_layer_data_v2(cfg);

    let mut pattern = compute_stack_raw(&layer_data, n, extent, cfg.honeycomb, displacement, None);
    normalize_unit(&mut pattern);

    let moire_period = moire_period_from_g_shells(&layer_data[0].0, &layer_data[1].0);

    Ok(GrapheneStackResult {
        pattern,
        resolution: n,
        physical_extent: extent,
        moire_period,
        n_layers: layer_data.len(),
    })
}

// ---------------------------------------------------------------------------
// Supermoire: graphene stack on a substrate overlayer
// ---------------------------------------------------------------------------

/// Configuration for a graphene stack on a substrate lattice: the intra-stack
/// moire beats against the graphene-substrate interface moire.
#[derive(Debug, Clone)]
pub struct SupermoireConfig {
    pub stack: GrapheneStackConfigV2,
    /// Substrate overlayer lattice constant in Angstrom.
    pub overlayer_a: f64,
    pub overlayer_lattice_type: LatticeType,
    /// Twist between the graphene stack and the overlayer in degrees.
    pub interface_twist_deg: f64,
}

impl Default for SupermoireConfig {
    fn default() -> Self {
        Self {
            stack: GrapheneStackConfigV2::default(),
            // Sb2Te3-like hexagonal substrate
            overlayer_a: 4.264,
            overlayer_lattice_type: LatticeType::Hexagonal,
            interface_twist_deg: 0.0,
        }
    }
}

/// Result of a supermoire computation.
#[derive(Debug, Clone)]
pub struct SupermoireResult {
    /// Row-major intensity values, resolution x resolution, normalized to [0, 1].
    pub pattern: Vec<f64>,
    pub resolution: usize,
    pub physical_extent: f64,
    /// Intra-stack moire period in Angstroms (infinite for untwisted,
    /// unstrained stacks).
    pub stack_period: f64,
    /// Graphene-overlayer interface moire period in Angstroms.
    pub interface_period: f64,
    /// Beat period L1*L2 / |L1 - L2| of the two moire lengths in Angstroms.
    pub supermoire_period: f64,
    /// Stack layers plus the overlayer.
    pub n_layers: usize,
}

/// Reciprocal G-vectors of the substrate overlayer.  Mirrors the private
/// `reciprocal_g_vectors` in `moire.rs`.
fn overlayer_g_vectors(lattice_type: LatticeType, a: f64) -> Vec<Vector2<f64>> {
    match lattice_type {
        LatticeType::Square => {
            let g = 2.0 * PI / a;
            vec![
                Vector2::new(g, 0.0),
                Vector2::new(0.0, g),
                Vector2::new(-g, 0.0),
                Vector2::new(0.0, -g),
            ]
        }
        LatticeType::Hexagonal => hex_g_vectors(a).to_vec(),
    }
}

/// Beat period of two moire lengths: L1*L2 / |L1 - L2|, degrading gracefully
/// to the finite one when the other is infinite.
fn beat_period(l1: f64, l2: f64) -> f64 {
    if !l1.is_finite() {
        return l2;
    }
    if !l2.is_finite() {
        return l1;
    }
    let diff = (l1 - l2).abs();
    if diff < 1e-12 {
        return f64::INFINITY;
    }
    l1 * l2 / diff
}

/// Compute the supermoire pattern of a graphene stack on a substrate: the
/// raw stack potential is multiplied by the substrate plane-wave potential
/// and normalized to [0, 1].
pub fn compute_supermoire(cfg: &SupermoireConfig) -> Result<SupermoireResult, String> {
    validate_stack_config_v2(&cfg.stack, None)?;
    if cfg.overlayer_a <= 0.0 {
        return Err(format!(
            "overlayer_a must be positive, got {}",
            cfg.overlayer_a
        ));
    }

    let n = cfg.stack.base.resolution;
    let extent = cfg.stack.base.physical_extent;
    let layer_data = stack_layer_data_v2(&cfg.stack);

    let twist_rad = cfg.interface_twist_deg.to_radians();
    let (c, s) = (twist_rad.cos(), twist_rad.sin());
    let over_gs: Vec<Vector2<f64>> =
        overlayer_g_vectors(cfg.overlayer_lattice_type, cfg.overlayer_a)
            .iter()
            .map(|g| Vector2::new(g.x * c - g.y * s, g.x * s + g.y * c))
            .collect();

    let mut pattern = compute_stack_raw(
        &layer_data,
        n,
        extent,
        cfg.stack.honeycomb,
        None,
        Some(&over_gs),
    );
    normalize_unit(&mut pattern);

    let stack_period = moire_period_from_g_shells(&layer_data[0].0, &layer_data[1].0);

    // Interface period estimate, matching the branch logic of moire.rs.
    let a_g = cfg.stack.base.lattice_a;
    let interface_period = if cfg.interface_twist_deg.abs() < 1e-6 {
        moire_periodicity_1d(a_g, cfg.overlayer_a)
    } else if (a_g - cfg.overlayer_a).abs() < 1e-12 {
        moire_periodicity_twist(a_g, cfg.interface_twist_deg)
    } else {
        moire_periodicity_1d(a_g, cfg.overlayer_a).min(moire_periodicity_twist(
            (a_g + cfg.overlayer_a) / 2.0,
            cfg.interface_twist_deg,
        ))
    };

    Ok(SupermoireResult {
        pattern,
        resolution: n,
        physical_extent: extent,
        stack_period,
        interface_period,
        supermoire_period: beat_period(stack_period, interface_period),
        n_layers: layer_data.len() + 1,
    })
}

// ---------------------------------------------------------------------------
// Bistritzer-MacDonald magic angle
// ---------------------------------------------------------------------------

/// First magic angle in degrees for an n-layer stack.
///
/// theta = 2 * asin(sqrt(3) * w_eff / (2 * hbar*v_F * k_D)) with
/// k_D = 4*pi / (3a).  Alternating-twist trilayers have w_eff = sqrt(2)*w
/// (Khalaf 2019); bilayers use the bare w.
pub fn magic_angle_deg(n_layers: usize) -> Result<f64, String> {
    let w_eff = match n_layers {
        2 => W_INTERLAYER_TBG,
        3 => 2.0_f64.sqrt() * W_INTERLAYER_TBG,
        _ => return Err(format!("n_layers must be 2 or 3, got {}", n_layers)),
    };
    let k_d = 4.0 * PI / (3.0 * GRAPHENE_A);
    let theta = 2.0 * (3.0_f64.sqrt() * w_eff / (2.0 * HBAR_VF_GRAPHENE * k_d)).asin();
    Ok(theta.to_degrees())
}

/// Renormalized Dirac velocity ratio v*/v_F at a twist angle.
///
/// v*/v_F = (1 - 3*alpha^2) / (1 + 6*alpha^2) with
/// alpha = w / (hbar*v_F * k_theta), k_theta = 2*k_D*sin(theta/2)
/// (Bistritzer-MacDonald first magic-angle series).
/// Valid for twist_angle_deg > 0 — alpha diverges as theta -> 0.
pub fn dirac_velocity_ratio(twist_angle_deg: f64) -> f64 {
    let k_d = 4.0 * PI / (3.0 * GRAPHENE_A);
    let k_theta = 2.0 * k_d * (twist_angle_deg.to_radians() / 2.0).sin();
    let alpha = W_INTERLAYER_TBG / (HBAR_VF_GRAPHENE * k_theta);
    (1.0 - 3.0 * alpha * alpha) / (1.0 + 6.0 * alpha * alpha)
}

// ---------------------------------------------------------------------------
// SPECULATIVE — flat-band superconductivity
// ---------------------------------------------------------------------------

/// Configuration for flat-band superconductivity estimation.
#[derive(Debug, Clone)]
pub struct FlatBandConfig {
    /// Twist angle in degrees (must be positive).
    pub twist_angle_deg: f64,
    /// Number of layers: 2 (TBG) or 3 (alternating-twist trilayer).
    pub n_layers: usize,
    /// Moire-band filling nu in [-4, 4].
    pub filling: f64,
}

impl Default for FlatBandConfig {
    fn default() -> Self {
        Self {
            twist_angle_deg: 1.08,
            n_layers: 2,
            filling: 2.4,
        }
    }
}

/// Result of a flat-band superconductivity estimation.
#[derive(Debug, Clone)]
pub struct FlatBandResult {
    /// Dimensionless BM coupling alpha = w / (hbar*v_F * k_theta).
    pub alpha: f64,
    /// Renormalized Dirac velocity ratio v*/v_F.
    pub velocity_ratio: f64,
    /// First magic angle for this layer count in degrees.
    pub theta_magic_deg: f64,
    /// **SPECULATIVE**: estimated superconducting gap in meV.
    pub delta_mev: f64,
    /// **SPECULATIVE**: BCS critical temperature in Kelvin.
    pub tc_kelvin: f64,
    /// **SPECULATIVE**: filling-dome suppression factor in [0, 1].
    pub dome_factor: f64,
}

/// **SPECULATIVE**: Estimate flat-band superconductivity for a twisted stack.
///
/// Gap = Delta_max * Lorentzian(theta; theta_magic, width) * dome(|nu|),
/// dome(|nu|) = max(0, 1 - ((|nu| - nu_opt) / w_nu)^2); Tc from the
/// weak-coupling BCS ratio Delta = 1.764 * k_B * Tc.  Peak values are
/// anchored to Cao 2018 (TBG) and Park 2021 (trilayer).
pub fn compute_flat_band_sc(cfg: &FlatBandConfig) -> Result<FlatBandResult, String> {
    if cfg.twist_angle_deg <= 0.0 {
        return Err(format!(
            "twist_angle_deg must be positive, got {}",
            cfg.twist_angle_deg
        ));
    }
    if cfg.filling.abs() > 4.0 {
        return Err(format!(
            "filling must satisfy |nu| <= 4, got {}",
            cfg.filling
        ));
    }
    let theta_magic = magic_angle_deg(cfg.n_layers)?;

    let k_d = 4.0 * PI / (3.0 * GRAPHENE_A);
    let k_theta = 2.0 * k_d * (cfg.twist_angle_deg.to_radians() / 2.0).sin();
    let alpha = W_INTERLAYER_TBG / (HBAR_VF_GRAPHENE * k_theta);
    let velocity_ratio = dirac_velocity_ratio(cfg.twist_angle_deg);

    let detuning = (cfg.twist_angle_deg - theta_magic) / THETA_SC_WIDTH_DEG;
    let lorentz = 1.0 / (1.0 + detuning * detuning);

    let nu_detuning = (cfg.filling.abs() - NU_OPTIMAL_FILLING) / NU_DOME_WIDTH;
    let dome_factor = (1.0 - nu_detuning * nu_detuning).max(0.0);

    let delta_max = if cfg.n_layers == 2 {
        DELTA_TBG_MAX
    } else {
        DELTA_TTG_MAX
    };
    let delta_mev = lorentz * dome_factor * delta_max;
    let tc_kelvin = delta_mev * 1e-3 / (BCS_GAP_RATIO * KB_EV_K);

    Ok(FlatBandResult {
        alpha,
        velocity_ratio,
        theta_magic_deg: theta_magic,
        delta_mev,
        tc_kelvin,
        dome_factor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_layer_counts() {
        for kind in StackingKind::ALL {
            let layers = stack_layers(kind, 1.08);
            assert_eq!(layers.len(), kind.n_layers(), "kind {:?}", kind);
            let has_twist = layers.iter().any(|l| l.twist_deg.abs() > 1e-12);
            assert_eq!(has_twist, kind.is_twisted(), "kind {:?}", kind);
        }
    }

    #[test]
    fn test_registry_phase_at_origin() {
        // V(0) = sum_n cos(-c_n * idx * 2*pi/3): full constructive sum for
        // AA registry, exact cancellation for AB registry.
        let v_at_origin = |idx: f64| -> f64 {
            HEX_PHASE_COEFFS
                .iter()
                .map(|c| (-c * idx * 2.0 * PI / 3.0).cos())
                .sum()
        };
        assert!((v_at_origin(0.0) - 6.0).abs() < 1e-10);
        assert!(v_at_origin(1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pattern_normalized_and_finite() {
        let cfg = GrapheneStackConfig::default();
        let result = compute_graphene_stack(&cfg).unwrap();
        assert_eq!(result.pattern.len(), cfg.resolution * cfg.resolution);
        assert!(result.pattern.iter().all(|v| v.is_finite()));
        let min_val = result.pattern.iter().cloned().fold(f64::MAX, f64::min);
        let max_val = result.pattern.iter().cloned().fold(f64::MIN, f64::max);
        assert!(min_val >= -1e-10);
        assert!(max_val <= 1.0 + 1e-10);
    }

    #[test]
    fn test_aa_vs_ab_differ() {
        let cfg = |stacking| GrapheneStackConfig {
            stacking,
            twist_angle_deg: 0.0,
            resolution: 128,
            physical_extent: 30.0,
            ..Default::default()
        };
        let aa = compute_graphene_stack(&cfg(StackingKind::AA)).unwrap();
        let ab = compute_graphene_stack(&cfg(StackingKind::AB)).unwrap();
        let max_diff = aa
            .pattern
            .iter()
            .zip(ab.pattern.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff > 0.05, "max diff = {}", max_diff);
    }

    #[test]
    fn test_aba_vs_abc_differ() {
        let cfg = |stacking| GrapheneStackConfig {
            stacking,
            twist_angle_deg: 0.0,
            resolution: 128,
            physical_extent: 30.0,
            ..Default::default()
        };
        let aba = compute_graphene_stack(&cfg(StackingKind::ABA)).unwrap();
        let abc = compute_graphene_stack(&cfg(StackingKind::ABC)).unwrap();
        let max_diff = aba
            .pattern
            .iter()
            .zip(abc.pattern.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff > 0.05, "max diff = {}", max_diff);
    }

    #[test]
    fn test_tbg_stack_period() {
        let cfg = GrapheneStackConfig {
            stacking: StackingKind::TwistedBilayer,
            twist_angle_deg: 1.08,
            resolution: 32,
            ..Default::default()
        };
        let result = compute_graphene_stack(&cfg).unwrap();
        let expected = 2.46 / (2.0 * (0.54_f64.to_radians()).sin());
        assert!((expected - 130.51).abs() / 130.51 < 1e-3);
        let rel_err = (result.moire_period - expected).abs() / expected;
        assert!(rel_err < 1e-9, "period = {}", result.moire_period);
    }

    #[test]
    fn test_untwisted_stack_infinite_period() {
        let cfg = GrapheneStackConfig {
            stacking: StackingKind::AA,
            resolution: 32,
            ..Default::default()
        };
        let result = compute_graphene_stack(&cfg).unwrap();
        assert!(result.moire_period.is_infinite());
    }

    #[test]
    fn test_magic_angle_bilayer() {
        let theta = magic_angle_deg(2).unwrap();
        assert!(theta > 1.0 && theta < 1.2, "theta = {}", theta);
        assert!((theta - 1.076).abs() < 0.01, "theta = {}", theta);
    }

    #[test]
    fn test_magic_angle_trilayer_sqrt2_enhancement() {
        let theta_2 = magic_angle_deg(2).unwrap();
        let theta_3 = magic_angle_deg(3).unwrap();
        let ratio = theta_3 / theta_2;
        assert!(
            (ratio / 2.0_f64.sqrt() - 1.0).abs() < 1e-3,
            "ratio = {}",
            ratio
        );
    }

    #[test]
    fn test_velocity_zero_at_magic_angle() {
        let theta = magic_angle_deg(2).unwrap();
        assert!(dirac_velocity_ratio(theta).abs() < 1e-9);
    }

    #[test]
    fn test_velocity_recovers_at_large_twist() {
        assert!(dirac_velocity_ratio(30.0) > 0.99);
    }

    #[test]
    fn test_gap_peak_at_magic_angle() {
        let cfg = FlatBandConfig {
            twist_angle_deg: magic_angle_deg(2).unwrap(),
            n_layers: 2,
            filling: 2.4,
        };
        let result = compute_flat_band_sc(&cfg).unwrap();
        assert!((result.delta_mev - DELTA_TBG_MAX).abs() < 1e-12);
    }

    #[test]
    fn test_tc_at_gap_peak() {
        let cfg = FlatBandConfig {
            twist_angle_deg: magic_angle_deg(2).unwrap(),
            n_layers: 2,
            filling: 2.4,
        };
        let result = compute_flat_band_sc(&cfg).unwrap();
        assert!(
            (result.tc_kelvin - 1.974).abs() < 0.02,
            "Tc = {}",
            result.tc_kelvin
        );
    }

    #[test]
    fn test_dome_zero_away_from_optimal_filling() {
        for filling in [0.0, 4.0] {
            let cfg = FlatBandConfig {
                filling,
                ..Default::default()
            };
            let result = compute_flat_band_sc(&cfg).unwrap();
            assert!(result.dome_factor.abs() < 1e-12, "filling = {}", filling);
            assert!(result.delta_mev.abs() < 1e-12, "filling = {}", filling);
        }
    }

    #[test]
    fn test_magic_angle_invalid_layers() {
        assert!(magic_angle_deg(4).is_err());
    }

    #[test]
    fn test_flat_band_invalid_inputs() {
        let with_layers = FlatBandConfig {
            n_layers: 4,
            ..Default::default()
        };
        assert!(compute_flat_band_sc(&with_layers).is_err());
        let with_filling = FlatBandConfig {
            filling: 5.0,
            ..Default::default()
        };
        assert!(compute_flat_band_sc(&with_filling).is_err());
        let with_twist = FlatBandConfig {
            twist_angle_deg: 0.0,
            ..Default::default()
        };
        assert!(compute_flat_band_sc(&with_twist).is_err());
    }

    #[test]
    fn test_honeycomb_layer_potential_at_origin() {
        // A registry at the origin: the A-sublattice sum is fully constructive
        // (6.0) and the B-sublattice phases cancel exactly.
        let v = layer_potential(0.0, 0.0, GRAPHENE_A, 0.0, 0, true);
        assert!((v - 6.0).abs() < 1e-10, "v = {}", v);
        let v_mono = layer_potential(0.0, 0.0, GRAPHENE_A, 0.0, 0, false);
        assert!((v_mono - 6.0).abs() < 1e-10, "v = {}", v_mono);
    }

    #[test]
    fn test_honeycomb_pattern_differs_from_monolayer_basis() {
        let base = GrapheneStackConfig {
            resolution: 64,
            physical_extent: 30.0,
            twist_angle_deg: 0.0,
            stacking: StackingKind::AA,
            ..Default::default()
        };
        let plain = compute_graphene_stack_v2(
            &GrapheneStackConfigV2 {
                base: base.clone(),
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let honeycomb = compute_graphene_stack_v2(
            &GrapheneStackConfigV2 {
                base,
                honeycomb: true,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        let max_diff = plain
            .pattern
            .iter()
            .zip(honeycomb.pattern.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff > 0.05, "max diff = {}", max_diff);
    }

    #[test]
    fn test_v2_defaults_bit_match_v1() {
        let base = GrapheneStackConfig {
            resolution: 96,
            ..Default::default()
        };
        let v1 = compute_graphene_stack(&base).unwrap();
        let v2 = compute_graphene_stack_v2(
            &GrapheneStackConfigV2 {
                base,
                ..Default::default()
            },
            None,
        )
        .unwrap();
        assert_eq!(v1.pattern, v2.pattern);
        assert_eq!(v1.resolution, v2.resolution);
        assert_eq!(v1.physical_extent, v2.physical_extent);
        assert_eq!(v1.n_layers, v2.n_layers);
        let rel = (v1.moire_period - v2.moire_period).abs() / v1.moire_period;
        assert!(rel < 1e-9, "v1 = {}, v2 = {}", v1.moire_period, v2.moire_period);
    }

    #[test]
    fn test_strained_g_vectors_zero_strain_identity() {
        let gs = hex_g_vectors(GRAPHENE_A);
        let strained = strained_g_vectors(&gs, 0.0, 37.0, POISSON_GRAPHENE);
        for (a, b) in gs.iter().zip(strained.iter()) {
            assert_eq!(a.x, b.x);
            assert_eq!(a.y, b.y);
        }
    }

    #[test]
    fn test_heterostrain_untwisted_aa_finite_period() {
        let cfg = GrapheneStackConfigV2 {
            base: GrapheneStackConfig {
                stacking: StackingKind::AA,
                twist_angle_deg: 0.0,
                resolution: 64,
                ..Default::default()
            },
            heterostrain_percent: 1.0,
            ..Default::default()
        };
        let strained = compute_graphene_stack_v2(&cfg, None).unwrap();
        assert!(
            strained.moire_period.is_finite() && strained.moire_period > 0.0,
            "period = {}",
            strained.moire_period
        );
        let unstrained = compute_graphene_stack_v2(
            &GrapheneStackConfigV2 {
                heterostrain_percent: 0.0,
                ..cfg
            },
            None,
        )
        .unwrap();
        assert!(unstrained.moire_period.is_infinite());
        let max_diff = strained
            .pattern
            .iter()
            .zip(unstrained.pattern.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff > 0.05, "max diff = {}", max_diff);
    }

    #[test]
    fn test_period_from_g_shells_twist_matches_analytic() {
        let gs = hex_g_vectors(GRAPHENE_A);
        let theta = 1.08_f64;
        let rotated = rotate_hex_g_vectors(&gs, theta.to_radians());
        let period = moire_period_from_g_shells(&gs, &rotated);
        let expected = GRAPHENE_A / (2.0 * (theta.to_radians() / 2.0).sin());
        let rel = (period - expected).abs() / expected;
        assert!(rel < 1e-9, "period = {}, expected = {}", period, expected);
    }

    #[test]
    fn test_period_from_g_shells_degenerate_infinite() {
        let gs = hex_g_vectors(GRAPHENE_A);
        assert!(moire_period_from_g_shells(&gs, &gs).is_infinite());
    }

    #[test]
    fn test_displacement_warp_changes_pattern() {
        let cfg = GrapheneStackConfigV2 {
            base: GrapheneStackConfig {
                resolution: 64,
                physical_extent: 100.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let n2 = 64 * 64;
        // Linear-ramp warp up to ~half a lattice constant.
        let u_x: Vec<f64> = (0..n2).map(|i| 1.2 * i as f64 / n2 as f64).collect();
        let u_y = vec![0.0; n2];
        let warped = compute_graphene_stack_v2(&cfg, Some((&u_x, &u_y))).unwrap();
        let flat = compute_graphene_stack_v2(&cfg, None).unwrap();
        let max_diff = warped
            .pattern
            .iter()
            .zip(flat.pattern.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_diff > 0.05, "max diff = {}", max_diff);
    }

    #[test]
    fn test_v2_invalid_inputs() {
        let with_strain = GrapheneStackConfigV2 {
            heterostrain_percent: 100.0,
            ..Default::default()
        };
        assert!(compute_graphene_stack_v2(&with_strain, None).is_err());
        let cfg = GrapheneStackConfigV2 {
            base: GrapheneStackConfig {
                resolution: 16,
                ..Default::default()
            },
            ..Default::default()
        };
        let too_short = vec![0.0; 16];
        assert!(compute_graphene_stack_v2(&cfg, Some((&too_short, &too_short))).is_err());
        let with_lattice = GrapheneStackConfigV2 {
            base: GrapheneStackConfig {
                lattice_a: 0.0,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(compute_graphene_stack_v2(&with_lattice, None).is_err());
    }

    #[test]
    fn test_supermoire_sane() {
        let cfg = SupermoireConfig {
            stack: GrapheneStackConfigV2 {
                base: GrapheneStackConfig {
                    resolution: 64,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let result = compute_supermoire(&cfg).unwrap();
        assert_eq!(result.pattern.len(), 64 * 64);
        assert!(result.pattern.iter().all(|v| v.is_finite()));
        let min_val = result.pattern.iter().cloned().fold(f64::MAX, f64::min);
        let max_val = result.pattern.iter().cloned().fold(f64::MIN, f64::max);
        assert!(min_val >= -1e-10 && max_val <= 1.0 + 1e-10);
        assert_eq!(result.n_layers, 3);
        // TBG stack period ~ 130.5 A, graphene/Sb2Te3 interface ~ 5.8 A.
        let expected_stack = 2.46 / (2.0 * (0.54_f64.to_radians()).sin());
        assert!((result.stack_period - expected_stack).abs() / expected_stack < 1e-9);
        let expected_interface = 2.46 * 4.264 / (4.264 - 2.46);
        assert!(
            (result.interface_period - expected_interface).abs() / expected_interface < 1e-9
        );
        let expected_beat = expected_stack * expected_interface
            / (expected_stack - expected_interface).abs();
        assert!(
            (result.supermoire_period - expected_beat).abs() / expected_beat < 1e-6,
            "supermoire = {}",
            result.supermoire_period
        );
    }

    #[test]
    fn test_supermoire_untwisted_stack_beat_is_interface() {
        let cfg = SupermoireConfig {
            stack: GrapheneStackConfigV2 {
                base: GrapheneStackConfig {
                    stacking: StackingKind::AA,
                    twist_angle_deg: 0.0,
                    resolution: 32,
                    ..Default::default()
                },
                ..Default::default()
            },
            ..Default::default()
        };
        let result = compute_supermoire(&cfg).unwrap();
        assert!(result.stack_period.is_infinite());
        assert!(result.interface_period.is_finite());
        assert!((result.supermoire_period - result.interface_period).abs() < 1e-12);
        let with_overlayer = SupermoireConfig {
            overlayer_a: 0.0,
            ..cfg
        };
        assert!(compute_supermoire(&with_overlayer).is_err());
    }

    #[test]
    fn test_stack_invalid_inputs() {
        let with_resolution = GrapheneStackConfig {
            resolution: 1,
            ..Default::default()
        };
        assert!(compute_graphene_stack(&with_resolution).is_err());
        let with_extent = GrapheneStackConfig {
            physical_extent: 0.0,
            ..Default::default()
        };
        assert!(compute_graphene_stack(&with_extent).is_err());
        let with_lattice = GrapheneStackConfig {
            lattice_a: 0.0,
            ..Default::default()
        };
        assert!(compute_graphene_stack(&with_lattice).is_err());
    }
}
