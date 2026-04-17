//! "About / Physics Reference" modal window. Content mirrors the Dash home
//! page About accordion so both UIs point at the same arxiv paper and the
//! same list of validated vs. speculative modules.

use egui::{Context, RichText};

pub fn show(ctx: &Context, open: &mut bool) {
    egui::Window::new("About — Moire CPDM Tool")
        .open(open)
        .default_width(540.0)
        .resizable(false)
        .collapsible(false)
        .show(ctx, |ui| {
            ui.heading("Moire Cooper-Pair Density Modulation");
            ui.add_space(4.0);
            ui.label(
                "This application visualizes moire superlattice patterns and \
                 the Cooper-pair density modulation (CPDM) induced in the \
                 proximity-coupled superconducting gap.",
            );

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Reference paper:");
                ui.hyperlink_to(
                    "arXiv:2602.22637",
                    "https://arxiv.org/abs/2602.22637",
                );
            });

            ui.add_space(10.0);
            ui.label(RichText::new("Validated modules").strong());
            ui.label("• Moire patterns (plane-wave superposition)");
            ui.label("• Gap modulation via BCS proximity");
            ui.label("• FFT power spectrum + peak detection");

            ui.add_space(8.0);
            ui.label(RichText::new("Speculative modules").strong());
            ui.label("• Isotope effects on gap / coherence length");
            ui.label("• Topological proximity & Majorana modes");
            ui.label("• Abrikosov vortex lattice + Zeeman/Pauli limits");

            ui.add_space(8.0);
            ui.label(
                RichText::new(
                    "Results from speculative modules are exploratory — treat \
                     them as illustrations, not quantitative predictions.",
                )
                .small()
                .italics(),
            );

            ui.add_space(12.0);
            ui.label(RichText::new("Keyboard shortcuts").strong());
            ui.label("  Ctrl+S        Save screenshot (PNG)");
            ui.label("  Ctrl+R        Reset parameters to defaults");
            ui.label("  R             Reset 3D camera");
            ui.label("  W             Toggle wireframe overlay");
            ui.label("  F1            Open this dialog");
        });
}
