use rayon::prelude::*;

use crate::reference::ReferenceOrbit;

pub struct IterationResult {
    /// Interleaved [iter_as_f32, mag2] per pixel, row-major
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Find the most interesting boundary region, balancing complexity with proximity to center.
/// Higher variance = more fractal detail (filaments, spirals vs smooth edges).
pub fn find_boundary_target(result: &IterationResult, _max_iter: u32) -> Option<(f64, f64)> {
    let w = result.width as usize;
    let h = result.height as usize;

    let grid = 16usize;
    let cell_w = w / grid;
    let cell_h = h / grid;

    let mut best_ndc_x = 0.0f64;
    let mut best_ndc_y = 0.0f64;
    let mut best_score = 0.0f64;
    let mut found = false;

    for gy in 0..grid {
        for gx in 0..grid {
            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            let mut count = 0u32;

            for py in (gy * cell_h)..((gy + 1) * cell_h) {
                for px in (gx * cell_w)..((gx + 1) * cell_w) {
                    let idx = (py * w + px) * 2;
                    let iter = result.data[idx] as f64;
                    sum += iter;
                    sum2 += iter * iter;
                    count += 1;
                }
            }

            if count == 0 {
                continue;
            }
            let mean = sum / count as f64;
            let variance = (sum2 / count as f64 - mean * mean).max(0.0);

            if variance < 1.0 {
                continue;
            }

            let ndc_x = ((gx as f64 + 0.5) / grid as f64) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((gy as f64 + 0.5) / grid as f64) * 2.0;
            let dist = ndc_x * ndc_x + ndc_y * ndc_y;

            // Prefer high variance (complex detail) with mild proximity bias
            let score = variance.sqrt() / (1.0 + dist * 2.0);

            if score > best_score {
                best_score = score;
                best_ndc_x = ndc_x;
                best_ndc_y = ndc_y;
                found = true;
            }
        }
    }

    if !found {
        return None;
    }

    Some((best_ndc_x, best_ndc_y))
}

/// Run perturbation iteration on CPU for all pixels in parallel.
pub fn compute(
    orbit: &ReferenceOrbit,
    width: u32,
    height: u32,
    zoom: f64,
    max_iter: u32,
) -> IterationResult {
    let aspect = width as f64 / height as f64;
    let scale = 2.0 / zoom;
    let n_pixels = (width * height) as usize;
    let orbit_len = orbit.z_re.len().min(max_iter as usize);

    let data: Vec<f32> = (0..n_pixels)
        .into_par_iter()
        .flat_map_iter(|idx| {
            let px = idx % width as usize;
            let py = idx / width as usize;

            // Map pixel to delta from center (in complex plane)
            let ndc_x = (px as f64 / width as f64) * 2.0 - 1.0;
            let ndc_y = 1.0 - (py as f64 / height as f64) * 2.0; // flip y

            let d0_re = ndc_x * aspect * scale;
            let d0_im = ndc_y * scale;

            let (iter, mag2) = perturbate_pixel(orbit, orbit_len, d0_re, d0_im, max_iter);

            [iter as f32, mag2]
        })
        .collect();

    IterationResult { data, width, height }
}

fn perturbate_pixel(
    orbit: &ReferenceOrbit,
    orbit_len: usize,
    d0_re: f64,
    d0_im: f64,
    max_iter: u32,
) -> (u32, f32) {
    let mut d_re = d0_re;
    let mut d_im = d0_im;

    for n in 0..orbit_len {
        let z_re = orbit.z_re[n];
        let z_im = orbit.z_im[n];

        // δ_{n+1} = 2·Z_n·δ_n + δ_n² + δ₀
        let new_d_re =
            2.0 * (z_re * d_re - z_im * d_im) + (d_re * d_re - d_im * d_im) + d0_re;
        let new_d_im =
            2.0 * (z_re * d_im + z_im * d_re) + 2.0 * d_re * d_im + d0_im;

        d_re = new_d_re;
        d_im = new_d_im;

        // Escape check: |Z_n+1 + δ_n+1|² > R²
        // Use Z_{n+1} if available, else Z_n + δ_n approximation
        let total_re;
        let total_im;
        if n + 1 < orbit_len {
            total_re = orbit.z_re[n + 1] + d_re;
            total_im = orbit.z_im[n + 1] + d_im;
        } else {
            total_re = z_re + d_re;
            total_im = z_im + d_im;
        }
        let mag2 = total_re * total_re + total_im * total_im;
        if mag2 > 256.0 {
            return (n as u32 + 1, mag2 as f32);
        }

        // Glitch detection + rebasing is Phase 6 — skip for now
    }

    (max_iter, 0.0)
}
