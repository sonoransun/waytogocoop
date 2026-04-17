//! Screenshot capture: regenerate the current 2D/3D view at target resolution
//! and encode it as PNG. Independent of the main render pipeline so the
//! on-screen view and the saved file stay visually identical.

use std::fs::File;
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use egui::ColorImage;

/// Write an RGBA `ColorImage` to a PNG file at `path`.
pub fn save_color_image_to_png(image: &ColorImage, path: &PathBuf) -> Result<(), String> {
    let [w, h] = image.size;
    let mut rgba: Vec<u8> = Vec::with_capacity(w * h * 4);
    for pixel in &image.pixels {
        let [r, g, b, a] = pixel.to_array();
        rgba.extend_from_slice(&[r, g, b, a]);
    }

    let file = File::create(path).map_err(|e| format!("create {path:?}: {e}"))?;
    let writer = BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, w as u32, h as u32);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let mut png_writer = encoder
        .write_header()
        .map_err(|e| format!("png header: {e}"))?;
    png_writer
        .write_image_data(&rgba)
        .map_err(|e| format!("png write: {e}"))?;
    Ok(())
}

/// Return a timestamped PNG path in the current working directory.
pub fn default_screenshot_path() -> PathBuf {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(format!("moire-screenshot-{ts}.png"))
}
