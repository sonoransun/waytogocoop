//! Top-of-window menu bar. Also owns the keyboard shortcut routing for
//! the same entries so the two stay in sync.

use egui::{Context, Key, Modifiers, TopBottomPanel};

use crate::app::MoireApp;

/// Actions a menu entry or keyboard shortcut can request. The `update` loop
/// applies them after the menu/input passes so borrows don't conflict.
#[derive(Debug, Clone, Copy)]
pub enum MenuAction {
    SaveScreenshot,
    ResetParameters,
    ResetCamera,
    ToggleWireframe,
    ToggleAxes,
    ToggleTheme,
    ShowAbout,
    Quit,
}

/// Render the menu bar. Returns any action the user triggered.
pub fn show(ctx: &Context, app: &MoireApp) -> Option<MenuAction> {
    let mut action = None;

    TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::menu::bar(ui, |ui: &mut egui::Ui| {
            ui.menu_button("File", |ui: &mut egui::Ui| {
                if ui.button("Save screenshot…    Ctrl+S").clicked() {
                    action = Some(MenuAction::SaveScreenshot);
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Quit").clicked() {
                    action = Some(MenuAction::Quit);
                    ui.close_menu();
                }
            });

            ui.menu_button("Edit", |ui: &mut egui::Ui| {
                if ui.button("Reset parameters    Ctrl+R").clicked() {
                    action = Some(MenuAction::ResetParameters);
                    ui.close_menu();
                }
            });

            ui.menu_button("View", |ui: &mut egui::Ui| {
                if ui.button("Reset 3D camera    R").clicked() {
                    action = Some(MenuAction::ResetCamera);
                    ui.close_menu();
                }
                let wireframe_label = if app.show_wireframe {
                    "✓ Wireframe    W"
                } else {
                    "  Wireframe    W"
                };
                if ui.button(wireframe_label).clicked() {
                    action = Some(MenuAction::ToggleWireframe);
                    ui.close_menu();
                }
                let axes_label = if app.show_world_axes {
                    "✓ World axes + scale bar"
                } else {
                    "  World axes + scale bar"
                };
                if ui.button(axes_label).clicked() {
                    action = Some(MenuAction::ToggleAxes);
                    ui.close_menu();
                }
                ui.separator();
                let theme_label = if app.dark_mode {
                    "☾ Switch to light"
                } else {
                    "☀ Switch to dark"
                };
                if ui.button(theme_label).clicked() {
                    action = Some(MenuAction::ToggleTheme);
                    ui.close_menu();
                }
            });

            ui.menu_button("Help", |ui: &mut egui::Ui| {
                if ui.button("About…    F1").clicked() {
                    action = Some(MenuAction::ShowAbout);
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("arXiv:2602.22637").clicked() {
                    ctx.open_url(egui::OpenUrl::new_tab("https://arxiv.org/abs/2602.22637"));
                    ui.close_menu();
                }
            });
        });
    });

    action
}

/// Inspect the input state for the keyboard shortcuts the menu advertises.
/// Returns the matched action (or `None`).
pub fn detect_shortcut(ctx: &Context) -> Option<MenuAction> {
    ctx.input(|i| {
        let cmd = Modifiers::COMMAND; // Ctrl on Linux/Windows, ⌘ on macOS.
        if i.modifiers.matches_logically(cmd) && i.key_pressed(Key::S) {
            return Some(MenuAction::SaveScreenshot);
        }
        if i.modifiers.matches_logically(cmd) && i.key_pressed(Key::R) {
            return Some(MenuAction::ResetParameters);
        }
        if i.modifiers.is_none() && i.key_pressed(Key::R) {
            return Some(MenuAction::ResetCamera);
        }
        if i.modifiers.is_none() && i.key_pressed(Key::W) {
            return Some(MenuAction::ToggleWireframe);
        }
        if i.key_pressed(Key::F1) {
            return Some(MenuAction::ShowAbout);
        }
        None
    })
}
