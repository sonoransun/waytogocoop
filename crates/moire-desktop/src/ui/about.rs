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
                ui.hyperlink_to("arXiv:2602.22637", "https://arxiv.org/abs/2602.22637");
            });

            // The two lists are the web home page's (pages/home.py); the
            // Speculative set is CLAUDE.md's: isotope effects, topological
            // (the whole Cooper 3D tab: `topological.rs` is a simplified 3D
            // extension, z-decay included), magnetic, spin fluctuations, PDW.
            ui.add_space(10.0);
            ui.label(RichText::new("Validated modules").strong());
            ui.label("• Moire patterns — plane-wave superposition");
            ui.label("• Gap modulation — BCS proximity with moire amplitude scaling");
            ui.label("• FFT analysis — peak detection over the power spectrum");

            ui.add_space(8.0);
            ui.label(RichText::new("Speculative modules").strong());
            ui.label(
                "• Isotope effects on the gap and coherence length (no direct \
                 Te-isotope data for FeTe; α = 0.4 is the corrected iron-based \
                 isotope-effect consensus, Khasanov et al., PRB 82, 212505 \
                 (2010)) — includes a HIGHLY SPECULATIVE exotic/synthetic-\
                 isotope mode (radioactive and hypothetical masses, what-if \
                 illustrations only)",
            );
            ui.label(
                "• Topological proximity / Majorana modes — 3D extensions of \
                 the 2D model (Cooper 3D tab: z-slice, decay profile and \
                 vortex-bound Majorana modes)",
            );
            ui.label(
                "• Abrikosov vortex lattice + Zeeman / Pauli limits — simplified \
                 models not validated for these heterostructures",
            );
            ui.label(
                "• Spin-fluctuation pairing — Boltzmann magnetotransport of \
                 electron-doped LCCO in which the anisotropic T-linear \
                 scattering tracks Tc (Duffy et al., Nat. Phys. 2026), plus an \
                 empirical pairing map and a moire glue-modulation what-if",
            );
            ui.label(
                "• Pair-density-wave modalities — Landau model of the PDW, its \
                 composite charge orders and their survival above Tc in UTe2 \
                 (Zhu et al., PNAS 2026), plus a moire-locked what-if for \
                 these heterostructures",
            );

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
