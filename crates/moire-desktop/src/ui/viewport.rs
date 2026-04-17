use egui::{Color32, Pos2, Rect, Ui};

use crate::app::{MoireApp, Tab, ViewMode};
use crate::render::axes::{self, AxesSpec, ColorbarSpec};

/// Camera rotation sensitivity (radians per pixel of drag).
const CAMERA_ROTATION_SENSITIVITY: f32 = 0.01;
/// Camera zoom sensitivity (distance per pixel of scroll).
const CAMERA_ZOOM_SENSITIVITY: f32 = 0.005;

/// Render the main visualization viewport with tab bar and 2D/3D toggle.
pub fn show_viewport(ui: &mut Ui, app: &mut MoireApp) {
    let prev_tab = app.active_tab;

    ui.horizontal(|ui| {
        ui.selectable_value(&mut app.active_tab, Tab::Pattern, "Moire Pattern");
        ui.selectable_value(&mut app.active_tab, Tab::Density, "Density Modulation");
        ui.selectable_value(&mut app.active_tab, Tab::Fourier, "Fourier Spectrum");
        ui.selectable_value(&mut app.active_tab, Tab::MagneticField, "Magnetic");
        ui.selectable_value(&mut app.active_tab, Tab::CooperSurface3D, "Cooper 3D");
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
            let max = d.gap_field.iter().copied().fold(f64::NEG_INFINITY, f64::max);
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
            colormap: moire_core::colormap::viridis,
            value_range: (0.0, 1.0),
            value_label: "intensity",
        },
        Tab::Density | Tab::CooperSurface3D => ViewMeta {
            x_range: (-half, half),
            y_range: (-half, half),
            x_label: "x (Å)",
            y_label: "y (Å)",
            colormap: moire_core::colormap::coolwarm,
            value_range: density_range(),
            value_label: "Δ (meV)",
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
                colormap: moire_core::colormap::inferno,
                value_range: (0.0, 1.0),
                value_label: "log₁₀(|F|²) norm",
            }
        }
        Tab::MagneticField => ViewMeta {
            x_range: (-half, half),
            y_range: (-half, half),
            x_label: "x (Å)",
            y_label: "y (Å)",
            colormap: moire_core::colormap::coolwarm,
            value_range: density_range(),
            value_label: "Δ (meV)",
        },
    })
}

fn show_flat_2d(ui: &mut Ui, app: &MoireApp) {
    let texture = match app.active_tab {
        Tab::Pattern => app.pattern_texture.as_ref(),
        Tab::Density => app.density_texture.as_ref(),
        Tab::Fourier => app.fft_texture.as_ref(),
        Tab::MagneticField => app.magnetic_texture.as_ref(),
        Tab::CooperSurface3D => app.density_texture.as_ref(),
    };

    let available = ui.available_rect_before_wrap();
    let (image_rect, colorbar_rect) = axes::layout(available);
    ui.allocate_rect(available, egui::Sense::hover());

    let Some(meta) = view_meta(app) else { return; };

    if let Some(tex) = texture {
        let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
        ui.painter()
            .image(tex.id(), image_rect, uv, Color32::WHITE);

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
        ui.painter().rect_filled(rect, 0.0, ui.visuals().extreme_bg_color);
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
