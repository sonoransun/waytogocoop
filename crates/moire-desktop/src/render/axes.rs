//! Screen-space overlays that annotate a 2D texture view with axes and a
//! colorbar. Drawn with `egui::Painter`; no GPU resources are required.
//!
//! The colormap palette convention matches `moire-core::colormap` and the
//! Python `figure_factory` module: viridis for unsigned scalars, coolwarm for
//! signed gap modulation, inferno for FFT power spectra.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect, Stroke, StrokeKind, Vec2};

/// Configuration for a 2D axis overlay drawn around an image rect.
pub struct AxesSpec<'a> {
    pub x_range: (f64, f64),
    pub y_range: (f64, f64),
    pub x_label: &'a str,
    pub y_label: &'a str,
    /// Number of tick marks per axis (including endpoints).
    pub ticks: usize,
}

/// Configuration for a vertical colorbar drawn alongside an image.
pub struct ColorbarSpec<'a> {
    pub colormap: fn(f64) -> [u8; 4],
    pub value_range: (f64, f64),
    pub label: &'a str,
    /// Number of tick labels (including endpoints).
    pub ticks: usize,
}

fn stroke_color(dark: bool) -> Color32 {
    if dark {
        Color32::from_gray(180)
    } else {
        Color32::from_gray(60)
    }
}

fn text_color(dark: bool) -> Color32 {
    if dark {
        Color32::from_gray(220)
    } else {
        Color32::from_gray(30)
    }
}

/// Reserve margins around an available rect for axis labels and a colorbar.
///
/// Returns (image_rect, colorbar_rect). Call this before painting the texture
/// so the texture lands inside `image_rect` and there is room for the overlay.
pub fn layout(available: Rect) -> (Rect, Rect) {
    const LEFT_MARGIN: f32 = 44.0;
    const BOTTOM_MARGIN: f32 = 28.0;
    const TOP_MARGIN: f32 = 8.0;
    const COLORBAR_GAP: f32 = 10.0;
    const COLORBAR_WIDTH: f32 = 18.0;
    const COLORBAR_LABEL_SPACE: f32 = 60.0;

    let right_reserved = COLORBAR_GAP + COLORBAR_WIDTH + COLORBAR_LABEL_SPACE;

    let plot_min = Pos2::new(available.min.x + LEFT_MARGIN, available.min.y + TOP_MARGIN);
    let plot_max = Pos2::new(
        available.max.x - right_reserved,
        available.max.y - BOTTOM_MARGIN,
    );

    let w = (plot_max.x - plot_min.x).max(20.0);
    let h = (plot_max.y - plot_min.y).max(20.0);
    let side = w.min(h);

    let image_rect = Rect::from_min_size(plot_min, Vec2::splat(side));

    let colorbar_rect = Rect::from_min_size(
        Pos2::new(image_rect.max.x + COLORBAR_GAP, image_rect.min.y),
        Vec2::new(COLORBAR_WIDTH, image_rect.height()),
    );

    (image_rect, colorbar_rect)
}

/// Format a tick value with reasonable precision for the displayed range.
fn format_tick(value: f64, range: f64) -> String {
    if range == 0.0 || !range.is_finite() {
        return format!("{value:.2}");
    }
    let abs_range = range.abs();
    if abs_range >= 100.0 {
        format!("{value:.0}")
    } else if abs_range >= 10.0 {
        format!("{value:.1}")
    } else if abs_range >= 1.0 {
        format!("{value:.2}")
    } else if abs_range >= 0.01 {
        format!("{value:.3}")
    } else {
        format!("{value:.2e}")
    }
}

/// Draw axis ticks and labels around an image rect.
pub fn draw_axes(painter: &Painter, image: Rect, spec: &AxesSpec, dark: bool) {
    let stroke = Stroke::new(1.0, stroke_color(dark));
    let tc = text_color(dark);
    let tick_len: f32 = 5.0;

    painter.rect_stroke(image, 0.0, stroke, StrokeKind::Outside);

    let (x0, x1) = spec.x_range;
    let (y0, y1) = spec.y_range;
    let x_range = x1 - x0;
    let y_range = y1 - y0;
    let n = spec.ticks.max(2);

    // X ticks + labels along bottom edge.
    for i in 0..n {
        let frac = i as f32 / (n - 1) as f32;
        let px = image.min.x + frac * image.width();
        let xv = x0 + (i as f64 / (n - 1) as f64) * x_range;

        painter.line_segment(
            [Pos2::new(px, image.max.y), Pos2::new(px, image.max.y + tick_len)],
            stroke,
        );
        painter.text(
            Pos2::new(px, image.max.y + tick_len + 2.0),
            Align2::CENTER_TOP,
            format_tick(xv, x_range),
            FontId::proportional(10.0),
            tc,
        );
    }

    // X axis label (below tick labels).
    painter.text(
        Pos2::new(image.center().x, image.max.y + tick_len + 16.0),
        Align2::CENTER_TOP,
        spec.x_label,
        FontId::proportional(11.0),
        tc,
    );

    // Y ticks + labels along left edge.
    for i in 0..n {
        let frac = i as f32 / (n - 1) as f32;
        // Y increases upward in data-space; screen Y is inverted.
        let py = image.max.y - frac * image.height();
        let yv = y0 + (i as f64 / (n - 1) as f64) * y_range;

        painter.line_segment(
            [Pos2::new(image.min.x - tick_len, py), Pos2::new(image.min.x, py)],
            stroke,
        );
        painter.text(
            Pos2::new(image.min.x - tick_len - 2.0, py),
            Align2::RIGHT_CENTER,
            format_tick(yv, y_range),
            FontId::proportional(10.0),
            tc,
        );
    }

    // Y axis label, rotated 90° would require egui::Shape::text; keep horizontal
    // above the plot to avoid reimplementing glyph rotation.
    painter.text(
        Pos2::new(image.min.x, image.min.y - 4.0),
        Align2::LEFT_BOTTOM,
        spec.y_label,
        FontId::proportional(11.0),
        tc,
    );
}

/// Paint a vertical colorbar gradient with ticks and a label.
pub fn draw_colorbar(painter: &Painter, rect: Rect, spec: &ColorbarSpec, dark: bool) {
    let stroke = Stroke::new(1.0, stroke_color(dark));
    let tc = text_color(dark);
    let slices: i32 = 96;

    // Paint gradient top = max, bottom = min.
    let h = rect.height();
    for i in 0..slices {
        let t = 1.0 - (i as f64 / (slices - 1) as f64);
        let [r, g, b, a] = (spec.colormap)(t);
        let slice_top = rect.min.y + (i as f32) * h / slices as f32;
        let slice_bot = rect.min.y + ((i + 1) as f32) * h / slices as f32;
        let slice_rect = Rect::from_min_max(
            Pos2::new(rect.min.x, slice_top),
            Pos2::new(rect.max.x, slice_bot),
        );
        painter.rect_filled(slice_rect, 0.0, Color32::from_rgba_premultiplied(r, g, b, a));
    }

    painter.rect_stroke(rect, 0.0, stroke, StrokeKind::Outside);

    let (v0, v1) = spec.value_range;
    let v_span = v1 - v0;
    let n = spec.ticks.max(2);
    let tick_len: f32 = 4.0;

    for i in 0..n {
        let frac = i as f32 / (n - 1) as f32;
        let py = rect.max.y - frac * rect.height();
        let value = v0 + (i as f64 / (n - 1) as f64) * v_span;

        painter.line_segment(
            [Pos2::new(rect.max.x, py), Pos2::new(rect.max.x + tick_len, py)],
            stroke,
        );
        painter.text(
            Pos2::new(rect.max.x + tick_len + 2.0, py),
            Align2::LEFT_CENTER,
            format_tick(value, v_span),
            FontId::proportional(10.0),
            tc,
        );
    }

    // Colorbar label above the bar.
    painter.text(
        Pos2::new(rect.center().x, rect.min.y - 4.0),
        Align2::CENTER_BOTTOM,
        spec.label,
        FontId::proportional(11.0),
        tc,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_reserves_space_for_colorbar() {
        let avail = Rect::from_min_size(Pos2::ZERO, Vec2::new(600.0, 400.0));
        let (img, bar) = layout(avail);
        assert!(img.max.x < bar.min.x);
        assert!(img.width() > 0.0 && img.height() > 0.0);
        assert!(bar.width() > 0.0 && bar.height() > 0.0);
    }

    #[test]
    fn format_tick_scales_with_range() {
        assert_eq!(format_tick(200.0, 500.0), "200");
        assert!(format_tick(0.123, 0.5).starts_with("0.12"));
        assert!(format_tick(1e-4, 1e-3).contains("e"));
    }
}
