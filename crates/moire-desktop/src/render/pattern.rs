use egui::{Color32, ColorImage, Context, TextureHandle, TextureOptions};

/// Create an egui texture from a data grid using the specified colormap.
///
/// `data` is a row-major slice of f64 values in [0, 1], with `resolution x resolution` elements.
/// `colormap` maps a f64 in [0, 1] to an [u8; 4] RGBA value.
///
/// Orientation: data row 0 becomes the texture's top row (v = 0). The
/// moire-core grids put y_min in row 0 (`y = (iy / (n - 1) - 0.5) * extent`;
/// the shifted FFT's row 0 is -k_nyq), while `axes::draw_axes` labels y
/// increasing upward, so a view that pairs this texture with those axes must
/// paint it with v flipped (`uv = (0, 1)-(1, 0)`) to show the grid right-side
/// up, as the web heatmaps and the 3D surface's `+y` axis do:
/// `ui/viewport.rs::paint_texture_view` paints it that way, and the app's 2D
/// screenshot (`app.rs::capture_current_view`) reverses the rows to match.
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
