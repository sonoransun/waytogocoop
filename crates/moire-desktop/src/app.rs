use eframe::egui;
use moire_core::density::{DensityConfig, DensityResult};
use moire_core::materials;
use moire_core::moire::{MoireConfig, MoireResult};

use crate::render;
use crate::ui;

/// Active visualization tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Pattern,
    Density,
    Fourier,
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

    /// Run computation and create 2D textures.
    fn recompute(&mut self, ctx: &egui::Context) {
        let substrate = self.substrate_material();
        let overlayer = self.overlayer_material();

        let config = MoireConfig {
            substrate_a: substrate.a,
            substrate_lattice_type: substrate.lattice_type,
            overlayer_a: overlayer.a,
            overlayer_lattice_type: overlayer.lattice_type,
            twist_angle_deg: self.twist_angle,
            resolution: self.resolution,
            physical_extent: self.physical_extent,
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
        let density_min = density.gap_field.iter().cloned().fold(f64::MAX, f64::min);
        let density_max = density.gap_field.iter().cloned().fold(f64::MIN, f64::max);
        let density_range = density_max - density_min;
        let density_norm: Vec<f64> = if density_range > 1e-15 {
            density
                .gap_field
                .iter()
                .map(|v| (v - density_min) / density_range)
                .collect()
        } else {
            vec![0.5; density.gap_field.len()]
        };

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
        let (data, n, colormap): (&[f64], usize, fn(f64) -> [u8; 4]) = match self.active_tab {
            Tab::Pattern => {
                if let Some(ref m) = self.moire_result {
                    (&m.pattern, m.resolution, moire_core::colormap::viridis)
                } else {
                    return;
                }
            }
            Tab::Density => {
                if let Some(ref d) = self.density_result {
                    // Normalize for colormap
                    let min = d.gap_field.iter().cloned().fold(f64::MAX, f64::min);
                    let max = d.gap_field.iter().cloned().fold(f64::MIN, f64::max);
                    let range = max - min;
                    if range < 1e-15 {
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
        };

        // Special handling for density (needs normalization)
        if self.active_tab == Tab::Density {
            if let Some(ref d) = self.density_result {
                let min = d.gap_field.iter().cloned().fold(f64::MAX, f64::min);
                let max = d.gap_field.iter().cloned().fold(f64::MIN, f64::max);
                let range = max - min;
                if range < 1e-15 {
                    return;
                }
                let norm: Vec<f64> = d.gap_field.iter().map(|v| (v - min) / range).collect();
                let img = render::surface3d::render_surface_3d(
                    &norm,
                    d.resolution,
                    512,
                    512,
                    &self.camera,
                    moire_core::colormap::coolwarm,
                );
                self.surface_texture =
                    Some(ctx.load_texture("surface_3d", img, egui::TextureOptions::LINEAR));
            }
            return;
        }

        if n == 0 {
            return;
        }

        let img =
            render::surface3d::render_surface_3d(data, n, 512, 512, &self.camera, colormap);
        self.surface_texture =
            Some(ctx.load_texture("surface_3d", img, egui::TextureOptions::LINEAR));
    }

    /// Compute and render comparison textures for all 3 overlayers.
    fn refresh_comparison(&mut self, ctx: &egui::Context) {
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
            };
            let moire = moire_core::moire::compute_moire(&config);

            if self.view_mode == ViewMode::Surface3D {
                let img = render::surface3d::render_surface_3d(
                    &moire.pattern,
                    comp_res,
                    256,
                    256,
                    &self.camera,
                    moire_core::colormap::viridis,
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
            let d_min = density.gap_field.iter().cloned().fold(f64::MAX, f64::min);
            let d_max = density.gap_field.iter().cloned().fold(f64::MIN, f64::max);
            let d_range = d_max - d_min;
            let d_norm: Vec<f64> = if d_range > 1e-15 {
                density.gap_field.iter().map(|v| (v - d_min) / d_range).collect()
            } else {
                vec![0.5; density.gap_field.len()]
            };

            if self.view_mode == ViewMode::Surface3D {
                let img = render::surface3d::render_surface_3d(
                    &d_norm,
                    comp_res,
                    256,
                    256,
                    &self.camera,
                    moire_core::colormap::coolwarm,
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
            self.needs_recompute = false;
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
