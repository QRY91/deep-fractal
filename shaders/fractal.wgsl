// Mandelbrot fractal shader — perturbation coloring
// Reads precomputed iteration data from CPU, applies palette

struct Uniforms {
    resolution: vec2<f32>,
    center_hi: vec2<f32>,
    center_lo: vec2<f32>,
    zoom: f32,
    max_iter: u32,
    color_shift: f32,
    mode: u32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var palette_tex: texture_1d<f32>;
@group(0) @binding(2) var palette_samp: sampler;
@group(0) @binding(3) var iter_tex: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(vi & 1u)) * 4.0 - 1.0;
    let y = f32(i32((vi >> 1u) & 1u)) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Map screen pixel to iteration texture (which may be smaller)
    let iter_size = textureDimensions(iter_tex);
    let screen_uv = in.position.xy / u.resolution;
    let tx = i32(screen_uv.x * f32(iter_size.x));
    let ty = i32(screen_uv.y * f32(iter_size.y));
    let data = textureLoad(iter_tex, vec2<i32>(tx, ty), 0);
    let iter = u32(data.r);
    let mag2 = data.g;

    if (iter >= u.max_iter) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let smooth_iter = f32(iter) + 1.0 - log2(max(log2(max(mag2, 1.0)), 1.0)) / 2.0;
    let t = fract(smooth_iter / 64.0 + u.color_shift);
    return textureSampleLevel(palette_tex, palette_samp, t, 0.0);
}
