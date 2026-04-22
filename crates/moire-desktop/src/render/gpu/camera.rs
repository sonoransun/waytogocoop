//! View / projection matrix construction for the GPU path.
//!
//! The existing software [`Camera3D`] stores spherical coordinates
//! (`yaw`, `pitch`, `distance`). For the GPU we need proper `view` and
//! `projection` 4x4 matrices, which we build via `nalgebra` (already a
//! workspace dependency through `moire-core`) — no `glam` dependency needed.

use nalgebra::{Matrix4, Perspective3, Point3, Vector3};

use crate::render::renderer3d::Camera3D;

/// Convert the spherical camera to a `(view * proj)` matrix plus eye position.
///
/// Matches the software rasterizer's implicit coordinate system so swapping
/// backends produces visually equivalent output:
///
/// - `eye = distance * (cos(pitch) * sin(yaw), cos(pitch) * cos(yaw), sin(pitch))`
/// - `origin = (0, 0, 0)`, `up = +z`
/// - `fovy = 2 * atan(1 / focal)` with `focal = 2.0` (≈ 53°)
/// - near / far = `0.01` / `50.0`
pub fn view_proj_matrix(cam: &Camera3D, aspect: f32) -> ([[f32; 4]; 4], [f32; 3]) {
    let yaw = cam.yaw;
    let pitch = cam.pitch;
    let dist = cam.distance;
    let eye = Point3::new(
        dist * pitch.cos() * yaw.sin(),
        dist * pitch.cos() * yaw.cos(),
        dist * pitch.sin(),
    );
    let origin = Point3::new(0.0_f32, 0.0, 0.0);
    let up = Vector3::new(0.0, 0.0, 1.0);

    let view: Matrix4<f32> = nalgebra::Isometry3::look_at_rh(&eye, &origin, &up).to_homogeneous();
    let fovy = 2.0_f32 * (1.0_f32 / 2.0_f32).atan();
    let proj = Perspective3::new(aspect, fovy, 0.01, 50.0).to_homogeneous();
    let vp = proj * view;

    // nalgebra stores column-major; wgpu's WGSL expects the same when passed
    // as a `mat4x4<f32>` uniform. Return as a `[[f32;4]; 4]` so `bytemuck`
    // can send it straight to the buffer.
    let mut out = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            out[c][r] = vp[(r, c)];
        }
    }
    (out, [eye.x, eye.y, eye.z])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_camera_sees_origin_near_center() {
        let cam = Camera3D::default();
        let (vp, eye) = view_proj_matrix(&cam, 1.0);
        // Origin in clip space: project (0,0,0,1) through vp. Expect w != 0 and
        // ndc ~ (0, 0, something in [-1, 1]).
        let col3 = Vector3::new(vp[3][0], vp[3][1], vp[3][2]);
        let w = vp[3][3];
        let ndc_x = col3.x / w;
        let ndc_y = col3.y / w;
        assert!(ndc_x.abs() < 0.3, "origin ndc_x = {}", ndc_x);
        assert!(ndc_y.abs() < 0.3, "origin ndc_y = {}", ndc_y);
        // Eye distance matches.
        let dist = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
        assert!((dist - cam.distance).abs() < 1.0e-5);
    }

    #[test]
    fn points_on_opposite_sides_project_to_opposite_signs() {
        let cam = Camera3D {
            yaw: 0.0,
            pitch: 0.0,
            distance: 3.0,
        };
        let (vp, _) = view_proj_matrix(&cam, 1.0);
        // With look_at_rh at (0, dist, 0) → origin, world +x and world -x sit
        // on opposite sides of the camera's view axis. Whichever side +x
        // projects to, -x must project to the opposite side.
        let project = |p: [f32; 4]| -> Option<f32> {
            let mut clip = [0.0_f32; 4];
            for r in 0..4 {
                for c in 0..4 {
                    clip[r] += vp[c][r] * p[c];
                }
            }
            let w = clip[3];
            if w.abs() < 1.0e-6 {
                None
            } else {
                Some(clip[0] / w)
            }
        };
        let plus_x = project([1.0, 0.0, 0.0, 1.0]);
        let minus_x = project([-1.0, 0.0, 0.0, 1.0]);
        if let (Some(a), Some(b)) = (plus_x, minus_x) {
            assert!(
                a.signum() != b.signum(),
                "+x and -x should land on opposite sides: +x={} / -x={}",
                a,
                b,
            );
            assert!(a.is_finite() && b.is_finite());
        } else {
            panic!("projection degenerate");
        }
    }
}
