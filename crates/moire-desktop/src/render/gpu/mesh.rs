//! GPU mesh construction for a regular-grid heightmap.
//!
//! Produces an indexed triangle grid with per-vertex normals computed from
//! 4-neighbour central differences at upload time. Indexed layout lets us
//! reuse the same vertex buffer across frames and only re-upload when the
//! underlying height data changes.

use bytemuck::{Pod, Zeroable};

/// One heightmap vertex as seen by `surface.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    /// Scalar value in `[0, 1]`, sampled as `colormap_lut[value, 0]` by the
    /// fragment shader.
    pub value: f32,
}

/// CPU-side mesh state, carried in [`crate::render::gpu::GpuRenderer`].
pub struct MeshState {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub grid_n: usize,
}

/// Build interleaved vertex + index buffers for an NxN heightmap normalised
/// to `[0, 1]`. The mesh spans `x, y ∈ [-1, 1]` with `z ∈ [0, 0.3]` so it
/// matches the software rasterizer's implicit coordinate system.
pub fn build_mesh_data(data: &[f64], n: usize) -> (Vec<Vertex>, Vec<u32>) {
    assert!(data.len() == n * n, "heightmap length must equal n*n");
    assert!(n >= 2, "grid must be at least 2x2");
    let n_f = n as f32;
    let step = 2.0 / (n_f - 1.0);
    let z_scale = 0.3_f32;

    // Height with a 1-sample halo for cheap central differences at the edges.
    let height = |i: usize, j: usize| -> f32 {
        let i = i.clamp(0, n - 1);
        let j = j.clamp(0, n - 1);
        data[i * n + j] as f32 * z_scale
    };

    let mut vertices = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let x = -1.0 + step * j as f32;
            let y = -1.0 + step * i as f32;
            let z = height(i, j);
            // 4-neighbour central differences produce smoothly interpolated
            // normals that read as a single shaded surface (vs. the flat
            // per-face shading of the CPU rasterizer).
            let dzdx = height(i, j + 1) - height(i, j.saturating_sub(1));
            let dzdy = height(i + 1, j) - height(i.saturating_sub(1), j);
            let tangent_x = [2.0 * step, 0.0, dzdx];
            let tangent_y = [0.0, 2.0 * step, dzdy];
            // n = tangent_x × tangent_y
            let nx = tangent_x[1] * tangent_y[2] - tangent_x[2] * tangent_y[1];
            let ny = tangent_x[2] * tangent_y[0] - tangent_x[0] * tangent_y[2];
            let nz = tangent_x[0] * tangent_y[1] - tangent_x[1] * tangent_y[0];
            let inv_len = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt().max(1.0e-9);
            vertices.push(Vertex {
                position: [x, y, z],
                normal: [nx * inv_len, ny * inv_len, nz * inv_len],
                value: data[i * n + j] as f32,
            });
        }
    }

    // Two triangles per cell: (tl, tr, br) + (tl, br, bl).
    let cells = (n - 1) * (n - 1);
    let mut indices = Vec::with_capacity(cells * 6);
    for i in 0..(n - 1) {
        for j in 0..(n - 1) {
            let tl = (i * n + j) as u32;
            let tr = tl + 1;
            let bl = ((i + 1) * n + j) as u32;
            let br = bl + 1;
            indices.push(tl);
            indices.push(tr);
            indices.push(br);
            indices.push(tl);
            indices.push(br);
            indices.push(bl);
        }
    }
    (vertices, indices)
}

/// Vertex attribute layout for `surface.wgsl`.
pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    static ATTRS: [wgpu::VertexAttribute; 3] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32,   // value
    ];
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &ATTRS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mesh_index_count_matches_cells() {
        let n = 8;
        let data: Vec<f64> = (0..n * n).map(|i| (i as f64) / (n * n) as f64).collect();
        let (_, indices) = build_mesh_data(&data, n);
        let cells = (n - 1) * (n - 1);
        assert_eq!(indices.len(), cells * 6);
    }

    #[test]
    fn mesh_vertex_count_matches_grid() {
        let n = 5;
        let data = vec![0.0_f64; n * n];
        let (verts, _) = build_mesh_data(&data, n);
        assert_eq!(verts.len(), n * n);
    }

    #[test]
    fn flat_heightmap_produces_up_normals() {
        let n = 4;
        let data = vec![0.5_f64; n * n];
        let (verts, _) = build_mesh_data(&data, n);
        for v in &verts {
            assert!(
                v.normal[2] > 0.99 && v.normal[0].abs() < 0.01 && v.normal[1].abs() < 0.01,
                "flat mesh normal should be +z, got {:?}",
                v.normal
            );
        }
    }
}
