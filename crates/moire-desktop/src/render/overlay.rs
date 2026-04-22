//! Overlay rendering for vortex core markers and vector field arrows.

use egui::ColorImage;

/// Draw filled circle markers at specified positions on an image.
///
/// `points` are in data coordinates; they are mapped to pixel coordinates
/// using the image dimensions and `physical_extent`.
pub fn overlay_scatter_points(
    image: &mut ColorImage,
    points: &[[f64; 2]],
    physical_extent: f64,
    color: [u8; 4],
    radius_px: usize,
) {
    if physical_extent <= 0.0 {
        return;
    }
    let w = image.width();
    let h = image.height();
    let half = physical_extent / 2.0;

    for &[px, py] in points {
        // Convert data coords to pixel coords
        let ix = ((px + half) / physical_extent * w as f64) as i32;
        let iy = ((py + half) / physical_extent * h as f64) as i32;
        // Invert y for image coordinates
        let iy = h as i32 - 1 - iy;

        let r = radius_px as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let px_x = ix + dx;
                    let px_y = iy + dy;
                    if px_x >= 0 && px_x < w as i32 && px_y >= 0 && px_y < h as i32 {
                        let idx = px_y as usize * w + px_x as usize;
                        image.pixels[idx] = egui::Color32::from_rgba_unmultiplied(
                            color[0], color[1], color[2], color[3],
                        );
                    }
                }
            }
        }
    }
}

/// Draw a small cross marker at specified positions.
pub fn overlay_cross_markers(
    image: &mut ColorImage,
    points: &[[f64; 2]],
    physical_extent: f64,
    color: [u8; 4],
    half_size: usize,
) {
    if physical_extent <= 0.0 {
        return;
    }
    let w = image.width();
    let h = image.height();
    let half = physical_extent / 2.0;
    let s = half_size as i32;

    let c = egui::Color32::from_rgba_unmultiplied(color[0], color[1], color[2], color[3]);

    for &[px, py] in points {
        let ix = ((px + half) / physical_extent * w as f64) as i32;
        let iy = h as i32 - 1 - ((py + half) / physical_extent * h as f64) as i32;

        // Horizontal arm
        for dx in -s..=s {
            let x = ix + dx;
            if x >= 0 && x < w as i32 && iy >= 0 && iy < h as i32 {
                image.pixels[iy as usize * w + x as usize] = c;
            }
        }
        // Vertical arm
        for dy in -s..=s {
            let y = iy + dy;
            if ix >= 0 && ix < w as i32 && y >= 0 && y < h as i32 {
                image.pixels[y as usize * w + ix as usize] = c;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_image(w: usize, h: usize) -> ColorImage {
        ColorImage::new([w, h], egui::Color32::BLACK)
    }

    #[test]
    fn test_scatter_marks_center() {
        let mut img = make_test_image(100, 100);
        let extent = 100.0;
        overlay_scatter_points(&mut img, &[[0.0, 0.0]], extent, [255, 0, 0, 255], 2);
        // Center pixel should be red-ish
        let center = img.pixels[50 * 100 + 50];
        assert_ne!(center, egui::Color32::BLACK);
    }

    #[test]
    fn test_cross_markers() {
        let mut img = make_test_image(100, 100);
        let extent = 100.0;
        overlay_cross_markers(&mut img, &[[0.0, 0.0]], extent, [0, 255, 0, 255], 3);
        // At least one pixel should be green (not all black)
        let green = egui::Color32::from_rgba_unmultiplied(0, 255, 0, 255);
        let has_green = img.pixels.iter().any(|&p| p == green);
        assert!(
            has_green,
            "No green pixels found after cross marker overlay"
        );
    }
}
