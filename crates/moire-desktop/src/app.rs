use eframe::egui;
use moire_core::density::{DensityConfig, DensityResult};
use moire_core::isotope_effects::{IsotopeEffects, IsotopeEffectsConfig};
use moire_core::isotopes::IsotopeConfig;
use moire_core::magnetic::{MagneticFieldConfig, VortexLatticeResult, ZeemanResult};
use moire_core::topological::ProximityConfig;
use moire_core::materials;
use moire_core::moire::{MoireConfig, MoireResult};

use crate::render;
use crate::ui;

/// Default BCS coherence length for FeTe (Angstrom).
const DEFAULT_COHERENCE_LENGTH: f64 = 20.0;

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

/// Active visualization tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Pattern,
    Density,
    Fourier,
    MagneticField,
    CooperSurface3D,
}

/// 2D flat or 3D surface view mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ViewMode {
    Flat2D,
    Surface3D,
}

impl Default for Tab {
    fn default() -> Self {
        Tab::Pattern
    }
}

impl Default for ViewMode {
    fn default() -> Self {
        ViewMode::Flat2D
    }
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
    /// BCS isotope exponent.
    #[serde(default = "default_isotope_alpha")]
    pub isotope_alpha: f64,

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
    /// Whether the comparison window is open.
    #[serde(skip)]
    pub show_comparison: bool,
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
    /// Z-slice index for proximity 3D view.
    #[serde(skip)]
    pub z_slice_index: usize,
}

fn default_dark_mode() -> bool {
    true
}

fn default_isotope_alpha() -> f64 {
    0.25
}

fn default_g_factor() -> f64 {
    30.0
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
            isotope_alpha: 0.4,
            magnetic_config: MagneticFieldConfig::default(),
            proximity_config: ProximityConfig::default(),
            g_factor: 30.0,
            show_vortices: false,
            show_majorana: false,
            dark_mode: true,
            active_tab: Tab::Pattern,
            needs_recompute: true,
            needs_surface_rerender: true,
            moire_result: None,
            density_result: None,
            fft_data: None,
            pattern_texture: None,
            density_texture: None,
            fft_texture: None,
            camera: render::surface3d::Camera3D::default(),
            surface_texture: None,
            show_comparison: false,
            comparison_textures: None,
            comparison_needs_refresh: false,
            isotope_effects: None,
            vortex_result: None,
            zeeman_result: None,
            needs_magnetic_recompute: true,
            magnetic_texture: None,
            z_slice_index: 0,
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
        materials::substrate()
    }

    /// Get the overlayer material based on current selection.
    pub fn overlayer_material(&self) -> &'static moire_core::materials::Material {
        let overlayers = materials::overlayers();
        &overlayers[self.overlayer_idx.min(overlayers.len() - 1)]
    }

    /// Build an IsotopeConfig from the current app state.
    fn isotope_config(&self) -> IsotopeConfig {
        IsotopeConfig {
            fe_mass: self.fe_mass_override,
            te_mass: self.te_mass_override,
            sb_mass: self.sb_mass_override,
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

        let moire = moire_core::moire::compute_moire(&config);
        let density =
            moire_core::density::compute_density_modulation(&moire, &self.density_config);
        let fft_data = moire_core::fft::compute_fft_2d(&moire.pattern, moire.resolution);

        // Create 2D textures
        self.pattern_texture = Some(render::pattern::create_texture(
            ctx,
            "moire_pattern",
            &moire.pattern,
            moire.resolution,
            moire_core::colormap::viridis,
        ));

        // Normalize density for colormap
        let density_norm: Vec<f64> = normalize_to_unit_range(&density.gap_field)
            .unwrap_or_else(|| vec![0.5; density.gap_field.len()]);

        self.density_texture = Some(render::pattern::create_texture(
            ctx,
            "density_map",
            &density_norm,
            density.resolution,
            moire_core::colormap::coolwarm,
        ));

        self.fft_texture = Some(render::pattern::create_texture(
            ctx,
            "fft_spectrum",
            &fft_data,
            moire.resolution,
            moire_core::colormap::inferno,
        ));

        self.fft_data = Some(fft_data);
        self.moire_result = Some(moire);
        self.density_result = Some(density);
        self.needs_surface_rerender = true;
        self.comparison_needs_refresh = true;
    }

    /// Re-render the 3D surface texture from cached data.
    fn rerender_surface(&mut self, ctx: &egui::Context) {
        let surface_bg = if self.dark_mode {
            egui::Color32::from_rgb(30, 30, 35)
        } else {
            egui::Color32::from_rgb(240, 240, 245)
        };

        let (data, n, colormap): (&[f64], usize, fn(f64) -> [u8; 4]) = match self.active_tab {
            Tab::Pattern => {
                if let Some(ref m) = self.moire_result {
                    (&m.pattern, m.resolution, moire_core::colormap::viridis)
                } else {
                    return;
                }
            }
            Tab::Density | Tab::CooperSurface3D => {
                if let Some(ref d) = self.density_result {
                    // Check if normalizable; actual rendering handled below
                    if normalize_to_unit_range(&d.gap_field).is_none() {
                        return;
                    }
                    // We need owned data for density normalization — handle below
                    (&[], 0, moire_core::colormap::coolwarm)
                } else {
                    return;
                }
            }
            Tab::Fourier => {
                if let Some(ref f) = self.fft_data {
                    if let Some(ref m) = self.moire_result {
                        (f.as_slice(), m.resolution, moire_core::colormap::inferno)
                    } else {
                        return;
                    }
                } else {
                    return;
                }
            }
            Tab::MagneticField => {
                // Use suppression-modulated density for 3D surface
                if let Some(ref vr) = self.vortex_result {
                    if let Some(ref d) = self.density_result {
                        let combined = moire_core::magnetic::combined_gap_with_vortices(
                            &d.gap_field,
                            &vr.suppression_field,
                        );
                        if let Some(norm) = normalize_to_unit_range(&combined) {
                            let img = render::surface3d::render_surface_3d_with_bg(
                                &norm,
                                d.resolution,
                                512, 512,
                                &self.camera,
                                moire_core::colormap::coolwarm,
                                surface_bg,
                            );
                            self.surface_texture =
                                Some(ctx.load_texture("surface_3d", img, egui::TextureOptions::LINEAR));
                        }
                        return;
                    }
                }
                return;
            }
        };

        // Special handling for density (needs normalization)
        if self.active_tab == Tab::Density || self.active_tab == Tab::CooperSurface3D {
            if let Some(ref d) = self.density_result {
                if let Some(norm) = normalize_to_unit_range(&d.gap_field) {
                    let img = render::surface3d::render_surface_3d_with_bg(
                        &norm,
                        d.resolution,
                        512,
                        512,
                        &self.camera,
                        moire_core::colormap::coolwarm,
                        surface_bg,
                    );
                    self.surface_texture =
                        Some(ctx.load_texture("surface_3d", img, egui::TextureOptions::LINEAR));
                }
            }
            return;
        }

        if n == 0 {
            return;
        }

        let img =
            render::surface3d::render_surface_3d_with_bg(data, n, 512, 512, &self.camera, colormap, surface_bg);
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

        // Combine gap with vortex suppression
        if let Some(ref density) = self.density_result {
            let combined = moire_core::magnetic::combined_gap_with_vortices(
                &density.gap_field,
                &vortex.suppression_field,
            );

            // Normalize for colormap
            let norm: Vec<f64> = normalize_to_unit_range(&combined)
                .unwrap_or_else(|| vec![0.5; combined.len()]);

            // Build ColorImage from colormapped data
            let pixels: Vec<egui::Color32> = norm
                .iter()
                .map(|&v| {
                    let [r, g, b, a] = moire_core::colormap::coolwarm(v);
                    egui::Color32::from_rgba_premultiplied(r, g, b, a)
                })
                .collect();

            let mut img = egui::ColorImage {
                size: [resolution, resolution],
                pixels,
            };

            // Overlay vortex markers if enabled
            if self.show_vortices && !vortex.vortex_positions.is_empty() {
                let marker_color = if self.dark_mode {
                    [255, 255, 255, 255]
                } else {
                    [0, 0, 0, 255]
                };
                render::overlay::overlay_cross_markers(
                    &mut img,
                    &vortex.vortex_positions,
                    extent,
                    marker_color,
                    3,
                );
            }

            self.magnetic_texture = Some(
                ctx.load_texture("magnetic_overlay", img, egui::TextureOptions::LINEAR),
            );
        }

        // Compute Zeeman
        let delta_avg = (self.density_config.delta_1 + self.density_config.delta_2) / 2.0;
        self.zeeman_result = Some(moire_core::magnetic::compute_zeeman(
            &self.magnetic_config,
            delta_avg,
        ));

        self.vortex_result = Some(vortex);
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
            let moire = moire_core::moire::compute_moire(&config);

            if self.view_mode == ViewMode::Surface3D {
                let img = render::surface3d::render_surface_3d_with_bg(
                    &moire.pattern,
                    comp_res,
                    256,
                    256,
                    &self.camera,
                    moire_core::colormap::viridis,
                    surface_bg,
                );
                textures
                    .push(ctx.load_texture("comp_moire", img, egui::TextureOptions::LINEAR));
            } else {
                textures.push(render::pattern::create_texture(
                    ctx,
                    "comp_moire",
                    &moire.pattern,
                    comp_res,
                    moire_core::colormap::viridis,
                ));
            }

            let density =
                moire_core::density::compute_density_modulation(&moire, &self.density_config);
            let d_norm: Vec<f64> = normalize_to_unit_range(&density.gap_field)
                .unwrap_or_else(|| vec![0.5; density.gap_field.len()]);

            if self.view_mode == ViewMode::Surface3D {
                let img = render::surface3d::render_surface_3d_with_bg(
                    &d_norm,
                    comp_res,
                    256,
                    256,
                    &self.camera,
                    moire_core::colormap::coolwarm,
                    surface_bg,
                );
                textures
                    .push(ctx.load_texture("comp_density", img, egui::TextureOptions::LINEAR));
            } else {
                textures.push(render::pattern::create_texture(
                    ctx,
                    "comp_density",
                    &d_norm,
                    comp_res,
                    moire_core::colormap::coolwarm,
                ));
            }
        }

        self.comparison_textures = Some(textures);
    }
}

impl eframe::App for MoireApp {
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, eframe::APP_KEY, self);
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.needs_recompute {
            self.recompute(ctx);
            self.needs_magnetic_recompute = true;
            self.needs_recompute = false;
        }

        if self.needs_magnetic_recompute {
            self.recompute_magnetic(ctx);
            self.needs_magnetic_recompute = false;
        }

        if self.needs_surface_rerender && self.view_mode == ViewMode::Surface3D {
            self.rerender_surface(ctx);
            self.needs_surface_rerender = false;
        }

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
                    let substrate = materials::substrate();

                    if let Some(ref textures) = self.comparison_textures {
                        ui.horizontal(|ui| {
                            for (i, mat) in overlayers.iter().enumerate() {
                                ui.vertical(|ui| {
                                    ui.heading(format!("{} / {}", mat.formula, substrate.formula));
                                    ui.label(format!(
                                        "Mismatch: {:.1}%",
                                        (mat.a - substrate.a).abs() / substrate.a * 100.0
                                    ));
                                    let period = moire_core::moire::moire_periodicity_1d(
                                        mat.a,
                                        substrate.a,
                                    );
                                    ui.label(format!("Period: {:.1} A", period));
                                    ui.add_space(4.0);

                                    let moire_idx = i * 2;
                                    let density_idx = i * 2 + 1;
                                    let tex_size = egui::vec2(220.0, 220.0);

                                    if moire_idx < textures.len() {
                                        ui.label("Moire Pattern:");
                                        ui.image(egui::load::SizedTexture::new(
                                            textures[moire_idx].id(),
                                            tex_size,
                                        ));
                                    }
                                    if density_idx < textures.len() {
                                        ui.label("Density:");
                                        ui.image(egui::load::SizedTexture::new(
                                            textures[density_idx].id(),
                                            tex_size,
                                        ));
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
    }
}
