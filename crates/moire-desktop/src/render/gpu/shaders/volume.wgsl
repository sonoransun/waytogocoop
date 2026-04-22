// Volumetric raymarch shader for the Cooper-pair density 3D volume.
//
// Stretch-goal pass described in CONTRIBUTING.md: fullscreen-triangle fragment
// shader that steps through a `texture_3d<f32>` along eye rays, accumulating
// `colormap(sample) * density * opacity` with a front-to-back alpha blend.
//
// This is the GPU counterpart to the Python `go.Volume` path in
// `src/waytogocoop/components/figure_factory.py::create_3d_volume`.
//
// Conservative WGSL so naga translates to DXIL/Metal/SPIR-V without quirks:
// avoids `textureSampleGrad`, relies on linear filtering, caps the step count
// at a constant to keep the translator's bounds-tracking happy.

struct VolumeUniforms {
    inv_view_proj: mat4x4<f32>,
    eye_xyz:       vec3<f32>,
    step_count:    u32,
    volume_min:    vec3<f32>,
    clip_z:        f32,
    volume_max:    vec3<f32>,
    max_opacity:   f32,
}

@group(0) @binding(0) var<uniform> uniforms: VolumeUniforms;
@group(0) @binding(1) var volume_tex:     texture_3d<f32>;
@group(0) @binding(2) var volume_samp:    sampler;
@group(0) @binding(3) var colormap_lut:   texture_2d<f32>;
@group(0) @binding(4) var colormap_samp:  sampler;

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOut {
    // Fullscreen triangle: (-1,-1), (3,-1), (-1,3).
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = positions[vid];
    var out: VertexOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

fn unproject(ndc: vec2<f32>, z: f32) -> vec3<f32> {
    let clip = vec4<f32>(ndc, z, 1.0);
    let world = uniforms.inv_view_proj * clip;
    return world.xyz / world.w;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Cast a ray from the eye through the NDC point toward z=1.
    let near = unproject(in.ndc, -1.0);
    let far  = unproject(in.ndc,  1.0);
    let dir  = normalize(far - near);

    let max_steps = min(uniforms.step_count, 256u);
    let step_len = length(far - near) / f32(max_steps);

    var pos = near;
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);

    for (var i: u32 = 0u; i < max_steps; i = i + 1u) {
        pos = pos + dir * step_len;
        if (pos.z > uniforms.clip_z) {
            continue;
        }
        // Normalise world-space position into [0,1]^3 for the texture sample.
        let uvw = (pos - uniforms.volume_min) / (uniforms.volume_max - uniforms.volume_min);
        if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) {
            continue;
        }
        let density = textureSample(volume_tex, volume_samp, uvw).r;
        let color = textureSample(colormap_lut, colormap_samp, vec2<f32>(density, 0.5)).rgb;
        // Front-to-back compositing.
        let alpha_sample = clamp(density * uniforms.max_opacity, 0.0, 1.0);
        let a_out = acc.a + (1.0 - acc.a) * alpha_sample;
        let c_out = acc.rgb + (1.0 - acc.a) * alpha_sample * color;
        acc = vec4<f32>(c_out, a_out);
        if (acc.a >= 0.99) {
            break;
        }
    }

    return acc;
}
