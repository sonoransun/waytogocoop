// Surface shader for the wgpu 3D renderer.
//
// Pipeline:
//   vertex  : transform world-space heightmap vertices + carry forward
//             normal, scalar value, and world position
//   fragment: Lambert diffuse + Blinn-Phong specular + Fresnel-Schlick,
//             colormap LUT lookup (256x1 texture), optional clip-plane
//             discard when world_pos.z > uniforms.clip_z.
//
// Kept conservative so the naga WGSL frontend translates cleanly to DXIL
// (Windows), Metal (macOS/iOS), and SPIR-V (Linux Vulkan).

struct Uniforms {
    view_proj:  mat4x4<f32>,
    eye_xyz:    vec3<f32>,
    clip_z:     f32,
    light_dir:  vec3<f32>,
    ambient:    f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var colormap_lut: texture_2d<f32>;
@group(0) @binding(2) var colormap_samp: sampler;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) value:    f32,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) normal:    vec3<f32>,
    @location(2) value:     f32,
}

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_pos  = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.world_pos = input.position;
    out.normal    = normalize(input.normal);
    out.value     = input.value;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Clip-plane discard: keep vertices below the user-chosen z.
    if (in.world_pos.z > uniforms.clip_z) {
        discard;
    }

    let base_color = textureSample(colormap_lut, colormap_samp, vec2<f32>(in.value, 0.5)).rgb;

    let n = normalize(in.normal);
    let l = normalize(uniforms.light_dir);
    let v = normalize(uniforms.eye_xyz - in.world_pos);
    let h = normalize(l + v);

    let lambert = max(dot(n, l), 0.0);
    let spec_term = pow(max(dot(n, h), 0.0), 48.0);

    // Fresnel-Schlick with F0 = 0.04 (typical for dielectrics).
    let f0 = 0.04;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - max(dot(n, v), 0.0), 5.0);

    let diffuse = base_color * (uniforms.ambient + 0.8 * lambert);
    let specular = vec3<f32>(1.0, 1.0, 1.0) * spec_term * fresnel;

    return vec4<f32>(diffuse + specular, 1.0);
}
