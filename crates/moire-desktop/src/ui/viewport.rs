use std::ops::RangeInclusive;

use egui::{Color32, Pos2, Rect, Ui};
use egui_plot::{GridInput, GridMark, Line, LineStyle, Plot, VLine};

use crate::app::{CooperView, GrapheneView, MagneticView, MoireApp, Tab, ViewMode};
use crate::render::axes::{self, AxesSpec, ColorbarSpec};

/// Camera rotation sensitivity (radians per pixel of drag).
const CAMERA_ROTATION_SENSITIVITY: f32 = 0.01;
/// Camera zoom sensitivity (distance per pixel of scroll).
const CAMERA_ZOOM_SENSITIVITY: f32 = 0.005;
/// Bands whose energies never enter this +/- window (meV) are not drawn, so
/// the default plot bounds stay focused on the flat bands.
const BAND_PLOT_WINDOW_MEV: f64 = 250.0;
/// Highlight color for the two flat bands and the DOS trace; readable on
/// both the dark and light themes.
const FLAT_BAND_COLOR: Color32 = Color32::from_rgb(255, 140, 0);

/// Render the main visualization viewport with tab bar and 2D/3D toggle.
pub fn show_viewport(ui: &mut Ui, app: &mut MoireApp) {
    let prev_tab = app.active_tab;

    ui.horizontal(|ui| {
        ui.selectable_value(&mut app.active_tab, Tab::Pattern, "Moire Pattern");
        ui.selectable_value(&mut app.active_tab, Tab::Density, "Density Modulation");
        ui.selectable_value(&mut app.active_tab, Tab::Fourier, "Fourier Spectrum");
        ui.selectable_value(&mut app.active_tab, Tab::MagneticField, "Magnetic");
        ui.selectable_value(&mut app.active_tab, Tab::CooperSurface3D, "Cooper 3D");
        ui.selectable_value(&mut app.active_tab, Tab::Graphene, "Graphene");
        ui.separator();
        let _ = ui.selectable_value(&mut app.view_mode, ViewMode::Flat2D, "2D");
        if ui
            .selectable_value(&mut app.view_mode, ViewMode::Surface3D, "3D")
            .changed()
        {
            app.needs_surface_rerender = true;
        }
    });

    if app.active_tab != prev_tab && app.view_mode == ViewMode::Surface3D {
        app.needs_surface_rerender = true;
    }

    ui.separator();

    // Bands and DOS are egui_plot line charts, not textures: route them past
    // the texture paths in either view mode, so the 2D/3D toggle is inert.
    if app.active_tab == Tab::Graphene {
        match app.graphene_view {
            GrapheneView::Bands => {
                show_band_plot(ui, app);
                return;
            }
            GrapheneView::Dos => {
                show_dos_plot(ui, app);
                return;
            }
            _ => {}
        }
    }

    // The Cooper decay profile is a line chart too; route it past both texture
    // paths so the 2D/3D toggle is inert.
    if app.active_tab == Tab::CooperSurface3D && app.cooper_view == CooperView::DecayProfile {
        show_decay_plot(ui, app);
        return;
    }

    match app.view_mode {
        ViewMode::Flat2D => show_flat_2d(ui, app),
        ViewMode::Surface3D => show_surface_3d(ui, app),
    }
}

/// Per-tab axis + colorbar metadata for the 2D view.
struct ViewMeta {
    x_range: (f64, f64),
    y_range: (f64, f64),
    x_label: &'static str,
    y_label: &'static str,
    colormap: fn(f64) -> [u8; 4],
    value_range: (f64, f64),
    value_label: &'static str,
}

fn view_meta(app: &MoireApp) -> Option<ViewMeta> {
    let extent = app
        .moire_result
        .as_ref()
        .map(|m| m.physical_extent)
        .unwrap_or(app.physical_extent);
    let half = extent / 2.0;
    let resolution = app
        .moire_result
        .as_ref()
        .map(|m| m.resolution)
        .unwrap_or(app.resolution);

    let density_range = || -> (f64, f64) {
        if let Some(ref d) = app.density_result {
            let min = d.gap_field.iter().copied().fold(f64::INFINITY, f64::min);
            let max = d
                .gap_field
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            if min.is_finite() && max.is_finite() && (max - min).abs() > 1e-12 {
                return (min, max);
            }
        }
        let avg = (app.density_config.delta_1 + app.density_config.delta_2) / 2.0;
        (avg - 0.5, avg + 0.5)
    };

    Some(match app.active_tab {
        Tab::Pattern => ViewMeta {
            x_range: (-half, half),
            y_range: (-half, half),
            x_label: "x (Å)",
            y_label: "y (Å)",
            colormap: app.cmap(moire_core::colormap::viridis),
            value_range: (0.0, 1.0),
            value_label: "intensity",
        },
        Tab::Density => ViewMeta {
            x_range: (-half, half),
            y_range: (-half, half),
            x_label: "x (Å)",
            y_label: "y (Å)",
            colormap: app.cmap(moire_core::colormap::coolwarm),
            value_range: density_range(),
            value_label: "Δ (meV)",
        },
        Tab::CooperSurface3D => match app.cooper_view {
            CooperView::Majorana => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::viridis),
                value_range: (0.0, 1.0),
                value_label: "|ψ|² (norm)",
            },
            // Gap views (InterfaceGap / ZSlice); DecayProfile is routed to a
            // plot before reaching here.
            _ => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::coolwarm),
                value_range: app.cooper_value_range().unwrap_or((0.0, 1.0)),
                value_label: "Δ(z) (meV)",
            },
        },
        Tab::Fourier => {
            let k_nyq = if extent > 0.0 {
                std::f64::consts::PI * resolution as f64 / extent
            } else {
                1.0
            };
            ViewMeta {
                x_range: (-k_nyq, k_nyq),
                y_range: (-k_nyq, k_nyq),
                x_label: "kx (1/Å)",
                y_label: "ky (1/Å)",
                colormap: app.cmap(moire_core::colormap::inferno),
                value_range: (0.0, 1.0),
                value_label: "log₁₀(|F|²) norm",
            }
        }
        Tab::MagneticField => match app.magnetic_view {
            MagneticView::Susceptibility => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::plasma),
                value_range: (-1.0, 0.0),
                value_label: "χ (norm)",
            },
            MagneticView::ScreeningCurrent => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::plasma),
                value_range: (0.0, 1.0),
                value_label: "|j| / j_max",
            },
            MagneticView::CombinedGap => ViewMeta {
                x_range: (-half, half),
                y_range: (-half, half),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap: app.cmap(moire_core::colormap::coolwarm),
                value_range: density_range(),
                value_label: "Δ (meV)",
            },
        },
        Tab::Graphene => {
            // Fourier lives in k-space; Bands / Dos bypass the texture
            // display entirely but still get sensible plot-shaped meta.
            match app.graphene_view {
                GrapheneView::Fourier => {
                    let (g_res, g_extent) = app
                        .graphene_result
                        .as_ref()
                        .map(|g| (g.resolution, g.physical_extent))
                        .unwrap_or((resolution, extent));
                    let k_nyq = if g_extent > 0.0 {
                        std::f64::consts::PI * g_res as f64 / g_extent
                    } else {
                        1.0
                    };
                    return Some(ViewMeta {
                        x_range: (-k_nyq, k_nyq),
                        y_range: (-k_nyq, k_nyq),
                        x_label: "kx (1/Å)",
                        y_label: "ky (1/Å)",
                        colormap: app.cmap(moire_core::colormap::inferno),
                        value_range: (0.0, 1.0),
                        value_label: "log₁₀(|F|²) norm",
                    });
                }
                GrapheneView::Bands => {
                    let k_max = app
                        .band_structure
                        .as_ref()
                        .and_then(|b| b.k_distances.last().copied())
                        .filter(|&k| k > 0.0)
                        .unwrap_or(1.0);
                    return Some(ViewMeta {
                        x_range: (0.0, k_max),
                        y_range: (-BAND_PLOT_WINDOW_MEV, BAND_PLOT_WINDOW_MEV),
                        x_label: "k-path (1/Å)",
                        y_label: "E (meV)",
                        colormap: app.cmap(moire_core::colormap::viridis),
                        value_range: (-BAND_PLOT_WINDOW_MEV, BAND_PLOT_WINDOW_MEV),
                        value_label: "E (meV)",
                    });
                }
                GrapheneView::Dos => {
                    let (e_min, e_max, d_max) = app
                        .dos_result
                        .as_ref()
                        .map(|d| {
                            (
                                d.energies_mev.first().copied().unwrap_or(-1.0),
                                d.energies_mev.last().copied().unwrap_or(1.0),
                                d.dos.iter().copied().fold(0.0_f64, f64::max).max(1e-12),
                            )
                        })
                        .unwrap_or((-1.0, 1.0, 1.0));
                    return Some(ViewMeta {
                        x_range: (e_min, e_max),
                        y_range: (0.0, d_max),
                        x_label: "E (meV)",
                        y_label: "DOS",
                        colormap: app.cmap(moire_core::colormap::viridis),
                        value_range: (0.0, d_max),
                        value_label: "DOS (meV⁻¹ k⁻¹)",
                    });
                }
                _ => {}
            }
            let ghalf = app
                .graphene_result
                .as_ref()
                .map(|g| g.physical_extent)
                .unwrap_or(extent)
                / 2.0;
            #[allow(clippy::type_complexity)]
            let (colormap, value_range, value_label): (fn(f64) -> [u8; 4], (f64, f64), &'static str) =
                match app.graphene_view {
                GrapheneView::Pattern => {
                    (app.cmap(moire_core::colormap::viridis), (0.0, 1.0), "intensity")
                }
                GrapheneView::GapMap => {
                    let range = app
                        .graphene_gap
                        .as_ref()
                        .and_then(|g| {
                            let min = g.iter().copied().fold(f64::INFINITY, f64::min);
                            let max = g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                            if min.is_finite() && max.is_finite() && (max - min).abs() > 1e-12 {
                                Some((min, max))
                            } else {
                                None
                            }
                        })
                        .unwrap_or((0.0, 1.0));
                    (app.cmap(moire_core::colormap::coolwarm), range, "Δ (meV)")
                }
                GrapheneView::PseudoField => {
                    let b_max = app
                        .curvature_result
                        .as_ref()
                        .map(|c| c.max_abs_field)
                        .filter(|&m| m > 1e-15)
                        .unwrap_or(1.0);
                    (app.cmap(moire_core::colormap::coolwarm), (-b_max, b_max), "B_ps (T)")
                }
                GrapheneView::Strain => {
                    let s_max = app
                        .curvature_result
                        .as_ref()
                        .map(|c| {
                            c.strain_xx
                                .iter()
                                .zip(&c.strain_yy)
                                .zip(&c.strain_xy)
                                .map(|((&xx, &yy), &xy)| {
                                    (xx * xx + yy * yy + 2.0 * xy * xy).sqrt()
                                })
                                .fold(0.0_f64, f64::max)
                        })
                        .filter(|&m| m > 0.0)
                        .unwrap_or(1.0);
                    (app.cmap(moire_core::colormap::viridis), (0.0, s_max), "|ε|")
                }
                // Handled by the early returns above.
                GrapheneView::Fourier | GrapheneView::Bands | GrapheneView::Dos => {
                    unreachable!()
                }
            };
            ViewMeta {
                x_range: (-ghalf, ghalf),
                y_range: (-ghalf, ghalf),
                x_label: "x (Å)",
                y_label: "y (Å)",
                colormap,
                value_range,
                value_label,
            }
        }
    })
}

fn show_flat_2d(ui: &mut Ui, app: &MoireApp) {
    let texture = match app.active_tab {
        Tab::Pattern => app.pattern_texture.as_ref(),
        Tab::Density => app.density_texture.as_ref(),
        Tab::Fourier => app.fft_texture.as_ref(),
        Tab::MagneticField => app.magnetic_texture.as_ref(),
        Tab::CooperSurface3D => app.cooper_texture.as_ref(),
        Tab::Graphene => app.graphene_texture.as_ref(),
    };

    // The Fourier tab tucks a collapsible peak table below the spectrum. Only
    // shrink the image when the table is open, so the spectrum stays near
    // full-height by default.
    if app.active_tab == Tab::Fourier {
        let full = ui.available_rect_before_wrap();
        let header_id = ui.make_persistent_id("fft_peaks_header");
        let state = egui::collapsing_header::CollapsingState::load_with_default_open(
            ui.ctx(),
            header_id,
            false,
        );
        let table_reserve = if state.is_open() {
            (full.height() * 0.4).clamp(160.0, 300.0)
        } else {
            30.0
        };
        let image_h = (full.height() - table_reserve).max(120.0);
        let image_area = Rect::from_min_size(full.min, egui::vec2(full.width(), image_h));
        ui.allocate_rect(image_area, egui::Sense::hover());
        paint_texture_view(ui, app, texture, image_area);
        show_fft_peak_table(ui, app, state, table_reserve);
        return;
    }

    let available = ui.available_rect_before_wrap();
    ui.allocate_rect(available, egui::Sense::hover());
    paint_texture_view(ui, app, texture, available);
}

/// Paint a texture view (image + axes + colorbar) into `area`, or a centered
/// "Computing..." placeholder when the texture is missing.
fn paint_texture_view(ui: &Ui, app: &MoireApp, texture: Option<&egui::TextureHandle>, area: Rect) {
    let (image_rect, colorbar_rect) = axes::layout(area);
    let Some(meta) = view_meta(app) else {
        return;
    };

    if let Some(tex) = texture {
        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        ui.painter().image(tex.id(), image_rect, uv, Color32::WHITE);

        let axes_spec = AxesSpec {
            x_range: meta.x_range,
            y_range: meta.y_range,
            x_label: meta.x_label,
            y_label: meta.y_label,
            ticks: 5,
        };
        axes::draw_axes(ui.painter(), image_rect, &axes_spec, app.dark_mode);

        let cb_spec = ColorbarSpec {
            colormap: meta.colormap,
            value_range: meta.value_range,
            label: meta.value_label,
            ticks: 5,
        };
        axes::draw_colorbar(ui.painter(), colorbar_rect, &cb_spec, app.dark_mode);
    } else {
        ui.painter().text(
            image_rect.center(),
            egui::Align2::CENTER_CENTER,
            "Computing...",
            egui::FontId::default(),
            ui.visuals().text_color(),
        );
    }
}

/// Collapsible table of the strongest FFT peaks (kx, ky, |k|, wavelength,
/// amplitude). Scrolls internally so it never overflows its reserved band.
fn show_fft_peak_table(
    ui: &mut Ui,
    app: &MoireApp,
    state: egui::collapsing_header::CollapsingState,
    reserve: f32,
) {
    state
        .show_header(ui, |ui| {
            ui.strong("FFT peaks (top 20)");
        })
        .body(|ui| {
            let peaks = app.fft_peaks.as_deref().unwrap_or(&[]);
            if peaks.is_empty() {
                ui.weak("No peaks above threshold.");
                return;
            }
            egui::ScrollArea::vertical()
                .max_height((reserve - 28.0).max(60.0))
                .show(ui, |ui| {
                    render_fft_table(ui, peaks);
                });
        });
}

fn render_fft_table(ui: &mut Ui, peaks: &[moire_core::fft::FftPeak]) {
    use egui_extras::{Column, TableBuilder};

    TableBuilder::new(ui)
        .striped(true)
        .columns(Column::auto(), 4)
        .column(Column::remainder())
        .header(18.0, |mut header| {
            for label in [
                "kx (1/Å)",
                "ky (1/Å)",
                "|k| (1/Å)",
                "λ = 2π/|k| (Å)",
                "amplitude",
            ] {
                header.col(|ui| {
                    ui.strong(label);
                });
            }
        })
        .body(|mut body| {
            for p in peaks {
                let k_mag = (p.kx * p.kx + p.ky * p.ky).sqrt();
                let lambda = if k_mag > 1e-12 {
                    2.0 * std::f64::consts::PI / k_mag
                } else {
                    f64::INFINITY
                };
                body.row(16.0, |mut row| {
                    row.col(|ui| {
                        ui.monospace(format!("{:.4}", p.kx));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{:.4}", p.ky));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{k_mag:.4}"));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{lambda:.1}"));
                    });
                    row.col(|ui| {
                        ui.monospace(format!("{:.4}", p.amplitude));
                    });
                });
            }
        });
}

fn show_surface_3d(ui: &mut Ui, app: &mut MoireApp) {
    let available = ui.available_size();
    let side = available.x.min(available.y);
    let size = egui::vec2(side, side);

    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());

    if response.dragged() {
        let delta = response.drag_delta();
        app.camera.yaw += delta.x * CAMERA_ROTATION_SENSITIVITY;
        app.camera.pitch =
            (app.camera.pitch + delta.y * CAMERA_ROTATION_SENSITIVITY).clamp(-1.4, 1.4);
        app.needs_surface_rerender = true;
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            app.camera.distance =
                (app.camera.distance - scroll * CAMERA_ZOOM_SENSITIVITY).clamp(1.0, 5.0);
            app.needs_surface_rerender = true;
        }
    }

    if let Some(ref tex) = app.surface_texture {
        let uv = Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
    } else {
        ui.painter()
            .rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Computing 3D view...",
            egui::FontId::default(),
            ui.visuals().text_color(),
        );
    }

    if response.hovered() {
        ui.painter().text(
            rect.left_bottom() + egui::vec2(8.0, -20.0),
            egui::Align2::LEFT_BOTTOM,
            "Drag to rotate \u{2022} Scroll to zoom",
            egui::FontId::proportional(12.0),
            ui.visuals().weak_text_color(),
        );
    }
}

/// Centered placeholder text while a plot's backing data is missing.
fn show_computing_message(ui: &mut Ui, message: &str) {
    let rect = ui.available_rect_before_wrap();
    ui.allocate_rect(rect, egui::Sense::hover());
    ui.painter().text(
        rect.center(),
        egui::Align2::CENTER_CENTER,
        message,
        egui::FontId::default(),
        ui.visuals().text_color(),
    );
}

/// Line color for remote (non-flat) bands and reference lines.
fn muted_line_color(dark_mode: bool) -> Color32 {
    if dark_mode {
        Color32::from_gray(110)
    } else {
        Color32::from_gray(160)
    }
}

/// Draw the BM band structure: one line per band entering the display
/// window, the two central (flat) bands highlighted, x-axis ticks at the
/// K / Γ / M / K' path corners, and a flat-bandwidth overlay label.
fn show_band_plot(ui: &mut Ui, app: &MoireApp) {
    let Some(bs) = app.band_structure.as_ref() else {
        show_computing_message(ui, "Computing band structure...");
        return;
    };
    let n_bands = bs.energies_mev.first().map_or(0, Vec::len);
    if bs.k_distances.is_empty() || n_bands < 2 {
        show_computing_message(ui, "Computing band structure...");
        return;
    }

    let flat_lo = n_bands / 2 - 1;
    let remote_color = muted_line_color(app.dark_mode);
    let k_span = bs
        .k_distances
        .last()
        .copied()
        .unwrap_or(1.0)
        .max(f64::MIN_POSITIVE);
    let tick_tol = k_span * 1e-3;
    let spacer_ticks = bs.tick_positions.clone();
    let format_ticks: Vec<(f64, &'static str)> = bs
        .tick_positions
        .iter()
        .copied()
        .zip(bs.tick_labels.iter().copied())
        .collect();

    let response = Plot::new("graphene_bands")
        .x_axis_label("k-path")
        .y_axis_label("E (meV)")
        .x_grid_spacer(move |_input: GridInput| {
            spacer_ticks
                .iter()
                .map(|&value| GridMark {
                    value,
                    step_size: k_span,
                })
                .collect()
        })
        .x_axis_formatter(move |mark: GridMark, _range: &RangeInclusive<f64>| {
            format_ticks
                .iter()
                .find(|(pos, _)| (mark.value - pos).abs() < tick_tol)
                .map(|(_, label)| (*label).to_string())
                .unwrap_or_default()
        })
        .show(ui, |plot_ui| {
            for band in 0..n_bands {
                let enters_window = bs
                    .energies_mev
                    .iter()
                    .any(|energies| energies[band].abs() <= BAND_PLOT_WINDOW_MEV);
                if !enters_window {
                    continue;
                }
                let points: Vec<[f64; 2]> = bs
                    .k_distances
                    .iter()
                    .zip(bs.energies_mev.iter())
                    .map(|(&k, energies)| [k, energies[band]])
                    .collect();
                let line = if band == flat_lo || band == flat_lo + 1 {
                    Line::new(points).color(FLAT_BAND_COLOR).width(2.5)
                } else {
                    Line::new(points).color(remote_color).width(1.0)
                };
                plot_ui.line(line);
            }
        });

    ui.painter().text(
        response.response.rect.left_top() + egui::vec2(56.0, 8.0),
        egui::Align2::LEFT_TOP,
        format!(
            "flat bandwidth {:.2} meV \u{2022} gap to remote {:.2} meV",
            bs.flat_bandwidth_mev, bs.flat_gap_mev
        ),
        egui::FontId::proportional(12.0),
        ui.visuals().strong_text_color(),
    );
}

/// Draw the BM density of states as a single line with a dashed vertical
/// reference at the charge-neutrality point E = 0.
fn show_dos_plot(ui: &mut Ui, app: &MoireApp) {
    let Some(dos) = app.dos_result.as_ref() else {
        show_computing_message(ui, "Computing density of states...");
        return;
    };
    let points: Vec<[f64; 2]> = dos
        .energies_mev
        .iter()
        .zip(dos.dos.iter())
        .map(|(&e, &d)| [e, d])
        .collect();
    let zero_color = muted_line_color(app.dark_mode);

    Plot::new("graphene_dos")
        .x_axis_label("E (meV)")
        .y_axis_label("DOS (meV⁻¹ k⁻¹)")
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui.vline(
                VLine::new(0.0)
                    .color(zero_color)
                    .style(LineStyle::dashed_loose()),
            );
            plot_ui.line(Line::new(points).color(FLAT_BAND_COLOR).width(2.0));
        });
}

/// Draw the proximity decay profile f(z): unity inside the superconductor
/// (z < 0), exponential decay into the TI beyond the interface. Dashed
/// verticals mark the interface (z = 0) and the coherence length xi_prox.
fn show_decay_plot(ui: &mut Ui, app: &MoireApp) {
    let (Some(z_coords), Some(decay)) =
        (app.cooper_z_coords.as_ref(), app.cooper_decay.as_ref())
    else {
        show_computing_message(ui, "Computing proximity decay...");
        return;
    };
    let points: Vec<[f64; 2]> = z_coords
        .iter()
        .zip(decay.iter())
        .map(|(&z, &f)| [z, f])
        .collect();
    let ref_color = muted_line_color(app.dark_mode);
    let xi = app.proximity_config.xi_prox;

    Plot::new("cooper_decay")
        .x_axis_label("z (Å)")
        .y_axis_label("f(z)")
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui.vline(
                VLine::new(0.0)
                    .color(ref_color)
                    .style(LineStyle::dashed_loose()),
            );
            plot_ui.vline(
                VLine::new(xi)
                    .color(ref_color)
                    .style(LineStyle::dashed_loose()),
            );
            plot_ui.line(Line::new(points).color(FLAT_BAND_COLOR).width(2.0));
        });
}
