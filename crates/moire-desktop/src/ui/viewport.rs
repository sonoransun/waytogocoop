use egui::{Color32, Ui};

use crate::app::{MoireApp, Tab, ViewMode};

/// Render the main visualization viewport with tab bar and 2D/3D toggle.
pub fn show_viewport(ui: &mut Ui, app: &mut MoireApp) {
    let prev_tab = app.active_tab;

    // Tab bar + view mode toggle
    ui.horizontal(|ui| {
        ui.selectable_value(&mut app.active_tab, Tab::Pattern, "Moire Pattern");
        ui.selectable_value(&mut app.active_tab, Tab::Density, "Density Modulation");
        ui.selectable_value(&mut app.active_tab, Tab::Fourier, "Fourier Spectrum");
        ui.separator();
        if ui
            .selectable_value(&mut app.view_mode, ViewMode::Flat2D, "2D")
            .changed()
        {
            // Switching to 2D doesn't need re-render
        }
        if ui
            .selectable_value(&mut app.view_mode, ViewMode::Surface3D, "3D")
            .changed()
        {
            app.needs_surface_rerender = true;
        }
    });

    // Tab change triggers surface re-render in 3D mode
    if app.active_tab != prev_tab && app.view_mode == ViewMode::Surface3D {
        app.needs_surface_rerender = true;
    }

    ui.separator();

    match app.view_mode {
        ViewMode::Flat2D => show_flat_2d(ui, app),
        ViewMode::Surface3D => show_surface_3d(ui, app),
    }
}

fn show_flat_2d(ui: &mut Ui, app: &MoireApp) {
    let texture = match app.active_tab {
        Tab::Pattern => app.pattern_texture.as_ref(),
        Tab::Density => app.density_texture.as_ref(),
        Tab::Fourier => app.fft_texture.as_ref(),
    };

    if let Some(tex) = texture {
        let available = ui.available_size();
        let side = available.x.min(available.y);
        let size = egui::vec2(side, side);
        ui.centered_and_justified(|ui| {
            ui.image(egui::load::SizedTexture::new(tex.id(), size));
        });
    } else {
        ui.centered_and_justified(|ui| {
            ui.label("Computing...");
        });
    }
}

fn show_surface_3d(ui: &mut Ui, app: &mut MoireApp) {
    let available = ui.available_size();
    let side = available.x.min(available.y);
    let size = egui::vec2(side, side);

    // Allocate interactive area
    let (rect, response) = ui.allocate_exact_size(size, egui::Sense::click_and_drag());

    // Handle mouse drag for camera rotation
    if response.dragged() {
        let delta = response.drag_delta();
        app.camera.yaw += delta.x * 0.01;
        app.camera.pitch = (app.camera.pitch + delta.y * 0.01).clamp(-1.4, 1.4);
        app.needs_surface_rerender = true;
    }

    // Handle scroll for zoom
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll.abs() > 0.1 {
            app.camera.distance = (app.camera.distance - scroll * 0.005).clamp(1.0, 5.0);
            app.needs_surface_rerender = true;
        }
    }

    // Paint the surface texture
    if let Some(ref tex) = app.surface_texture {
        let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
        ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
    } else {
        ui.painter().rect_filled(rect, 0.0, Color32::from_rgb(30, 30, 35));
        ui.painter().text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            "Computing 3D view...",
            egui::FontId::default(),
            Color32::WHITE,
        );
    }

    // Hint text
    if response.hovered() {
        ui.painter().text(
            rect.left_bottom() + egui::vec2(8.0, -20.0),
            egui::Align2::LEFT_BOTTOM,
            "Drag to rotate \u{2022} Scroll to zoom",
            egui::FontId::proportional(12.0),
            Color32::from_white_alpha(140),
        );
    }
}
