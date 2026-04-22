//! Screenshot readback: copy the offscreen color attachment to CPU memory.
//!
//! wgpu's `copy_texture_to_buffer` requires each row's byte count to be a
//! multiple of `COPY_BYTES_PER_ROW_ALIGNMENT` (256 bytes). We request a
//! padded buffer, wait for the map, and strip the padding when repacking
//! into an `egui::ColorImage`.

use egui::{Color32, ColorImage};

/// Reconstruct an `egui::ColorImage` from a flat BGRA/RGBA readback buffer.
///
/// `bytes_per_row` is the **padded** stride returned by wgpu; `width` /
/// `height` are the texture dimensions in pixels.
///
/// The function accepts RGBA or BGRA; pass `swizzle_bgra = true` if the
/// texture format reads back as BGRA (e.g. on some Vulkan backends).
pub fn repack_rgba_to_color_image(
    bytes: &[u8],
    width: usize,
    height: usize,
    bytes_per_row: usize,
    swizzle_bgra: bool,
) -> ColorImage {
    let mut pixels = Vec::with_capacity(width * height);
    for row in 0..height {
        let row_start = row * bytes_per_row;
        for col in 0..width {
            let off = row_start + col * 4;
            let (r, g, b, a) = if swizzle_bgra {
                (bytes[off + 2], bytes[off + 1], bytes[off], bytes[off + 3])
            } else {
                (bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3])
            };
            pixels.push(Color32::from_rgba_premultiplied(r, g, b, a));
        }
    }
    ColorImage {
        size: [width, height],
        pixels,
    }
}

/// wgpu's copy-buffer alignment requirement.
pub const COPY_ROW_ALIGNMENT: usize = 256;

/// Round `bytes` up to the next multiple of `COPY_ROW_ALIGNMENT`.
pub fn padded_row_bytes(unpadded: usize) -> usize {
    let rem = unpadded % COPY_ROW_ALIGNMENT;
    if rem == 0 {
        unpadded
    } else {
        unpadded + (COPY_ROW_ALIGNMENT - rem)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_row_bytes_rounds_up() {
        assert_eq!(padded_row_bytes(0), 0);
        assert_eq!(padded_row_bytes(256), 256);
        assert_eq!(padded_row_bytes(257), 512);
        assert_eq!(padded_row_bytes(4 * 300), 1280); // 1200 -> 1280
    }

    #[test]
    fn repack_rgba_roundtrip() {
        // 3x2 pixels, RGBA, bytes_per_row = 256 (padded from 12)
        let width = 3;
        let height = 2;
        let bytes_per_row = 256;
        let mut bytes = vec![0u8; bytes_per_row * height];
        // Fill each pixel with (row, col, 0, 255).
        for row in 0..height {
            for col in 0..width {
                let off = row * bytes_per_row + col * 4;
                bytes[off] = row as u8;
                bytes[off + 1] = col as u8;
                bytes[off + 2] = 0;
                bytes[off + 3] = 255;
            }
        }
        let img = repack_rgba_to_color_image(&bytes, width, height, bytes_per_row, false);
        assert_eq!(img.size, [3, 2]);
        assert_eq!(
            img.pixels[0],
            Color32::from_rgba_premultiplied(0, 0, 0, 255)
        );
        assert_eq!(
            img.pixels[3],
            Color32::from_rgba_premultiplied(1, 0, 0, 255)
        );
        assert_eq!(
            img.pixels[5],
            Color32::from_rgba_premultiplied(1, 2, 0, 255)
        );
    }
}
