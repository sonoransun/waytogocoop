use moire_desktop::app;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Good Job Coop! — Moire Visualization"),
        ..Default::default()
    };
    eframe::run_native(
        "moire-viz",
        options,
        Box::new(|cc| Ok(Box::new(app::MoireApp::new(cc)))),
    )
}
