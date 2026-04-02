use egui::{Color32, ColorImage, Context, TextureHandle, TextureOptions};

/// Create an egui texture from a data grid using the specified colormap.
///
/// `data` is a row-major slice of f64 values in [0, 1], with `resolution x resolution` elements.
/// `colormap` maps a f64 in [0, 1] to an [u8; 4] RGBA value.
pub fn create_texture(
    ctx: &Context,
    name: &str,
    data: &[f64],
    resolution: usize,
    colormap: fn(f64) -> [u8; 4],
) -> TextureHandle {
    let pixels: Vec<Color32> = data
        .iter()
        .map(|&v| {
            let [r, g, b, a] = colormap(v);
            Color32::from_rgba_premultiplied(r, g, b, a)
        })
        .collect();

    let image = ColorImage {
        size: [resolution, resolution],
        pixels,
    };

    ctx.load_texture(name, image, TextureOptions::LINEAR)
}
