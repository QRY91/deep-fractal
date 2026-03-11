use rayon::prelude::*;

use crate::reference::{ReferenceOrbit, SeriesApprox};

pub struct IterationResult {
    /// Interleaved [iter_as_f32, mag2] per pixel, row-major
    pub data: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub sa_skipped: u32,
}

/// Find the most interesting boundary region, balancing complexity with proximity to center.
/// Higher variance = more fractal detail (filaments, spirals vs smooth edges).
pub fn find_boundary_target(result: &IterationResult, max_iter: u32) -> Option<(f64, f64)> {
    let w = result.width as usize;
    let h = result.height as usize;

    let grid = 16usize;
    let cell_w = w / grid;
    let cell_h = h / grid;
    let sub = 4usize; // sub-cells per axis within each coarse cell
    let sub_w = cell_w / sub;
    let sub_h = cell_h / sub;

    if sub_w == 0 || sub_h == 0 {
        return None;
    }

    // Two-pass: prefer complex boundary cells, fall back to "steer toward set"
    let mut best_strict = (0.0f64, 0.0f64, 0.0f64);
    let mut best_fallback = (0.0f64, 0.0f64, 0.0f64);
    let mut found_strict = false;
    let mut found_fallback = false;

    for gy in 0..grid {
        for gx in 0..grid {
            let mut has_interior = false;
            let mut has_exterior = false;
            let mut active_subcells = 0u32;
            let mut cell_sum = 0.0f64;
            let mut cell_count = 0u32;

            for sy in 0..sub {
                for sx in 0..sub {
                    let mut sum = 0.0f64;
                    let mut sum2 = 0.0f64;
                    let mut cnt = 0u32;

                    let py_start = gy * cell_h + sy * sub_h;
                    let px_start = gx * cell_w + sx * sub_w;
                    for py in py_start..(py_start + sub_h) {
                        for px in px_start..(px_start + sub_w) {
                            let idx = (py * w + px) * 2;
                            let iter_val = result.data[idx];
                            let iter = iter_val as u32;
                            if iter >= max_iter {
                                has_interior = true;
                            } else {
                                has_exterior = true;
                            }
                            let v = iter_val as f64;
                            sum += v;
                            sum2 += v * v;
                            cnt += 1;
                        }
                    }

                    if cnt > 0 {
                        let mean = sum / cnt as f64;
                        let var = (sum2 / cnt as f64 - mean * mean).max(0.0);
                        if var > 1.0 {
                            active_subcells += 1;
                        }
                        cell_sum += sum;
                        cell_count += cnt;
                    }
                }
            }

            if cell_count == 0 {
                continue;
            }

            let ndc_x = ((gx as f64 + 0.5) / grid as f64) * 2.0 - 1.0;
            let ndc_y = 1.0 - ((gy as f64 + 0.5) / grid as f64) * 2.0;
            let dist = ndc_x * ndc_x + ndc_y * ndc_y;

            if has_interior && has_exterior && active_subcells >= 2 {
                // Complex boundary: score by number of active sub-cells (fractal complexity)
                let score = active_subcells as f64 / (1.0 + dist * 2.0);
                if score > best_strict.2 {
                    best_strict = (ndc_x, ndc_y, score);
                    found_strict = true;
                }
            } else if has_exterior {
                // Fallback: steer toward highest mean iteration (closest to set)
                let mean = cell_sum / cell_count as f64;
                let mean_score = mean / (1.0 + dist * 2.0);
                if mean_score > best_fallback.2 {
                    best_fallback = (ndc_x, ndc_y, mean_score);
                    found_fallback = true;
                }
            }
        }
    }

    if found_strict {
        Some((best_strict.0, best_strict.1))
    } else if found_fallback {
        Some((best_fallback.0, best_fallback.1))
    } else {
        None
    }
}

/// Run perturbation iteration on CPU for all pixels in parallel.
/// Uses series approximation to skip the bulk of iterations.
pub fn compute(
    orbit: &ReferenceOrbit,
    sa: &SeriesApprox,
    width: u32,
    height: u32,
    zoom: f64,
    max_iter: u32,
) -> IterationResult {
    let aspect = width as f64 / height as f64;
    let scale = 2.0 / zoom;
    let n_pixels = (width * height) as usize;
    let orbit_len = orbit.z_re.len().min(max_iter as usize);

    // Global skip point based on max delta (corner pixel)
    let max_d0_mag2 = (aspect * scale) * (aspect * scale) + scale * scale;
    let skip = sa.skip_point(max_d0_mag2).min(orbit_len.saturating_sub(1));

    let data: Vec<f32> = (0..n_pixels)
        .into_par_iter()
        .flat_map_iter(|idx| {
            let px = idx % width as usize;
            let py = idx / width as usize;

            // Map pixel to delta from center (in complex plane)
            let ndc_x = (px as f64 / width as f64) * 2.0 - 1.0;
            let ndc_y = 1.0 - (py as f64 / height as f64) * 2.0;

            let d0_re = ndc_x * aspect * scale;
            let d0_im = ndc_y * scale;

            let (iter, mag2) = if skip > 0 {
                perturbate_pixel_sa(orbit, sa, orbit_len, d0_re, d0_im, max_iter, skip)
            } else {
                perturbate_pixel(orbit, orbit_len, d0_re, d0_im, max_iter)
            };

            [iter as f32, mag2]
        })
        .collect();

    IterationResult { data, width, height, sa_skipped: skip as u32 }
}

/// Perturbation with series approximation: jump to skip point, then iterate.
fn perturbate_pixel_sa(
    orbit: &ReferenceOrbit,
    sa: &SeriesApprox,
    orbit_len: usize,
    d0_re: f64,
    d0_im: f64,
    max_iter: u32,
    skip: usize,
) -> (u32, f32) {
    // Evaluate SA polynomial: δ_skip = A·δ₀ + B·δ₀² + C·δ₀³
    let d02_re = d0_re * d0_re - d0_im * d0_im;
    let d02_im = 2.0 * d0_re * d0_im;
    let d03_re = d02_re * d0_re - d02_im * d0_im;
    let d03_im = d02_re * d0_im + d02_im * d0_re;

    let (ar, ai) = (sa.a_re[skip], sa.a_im[skip]);
    let (br, bi) = (sa.b_re[skip], sa.b_im[skip]);
    let (cr, ci) = (sa.c_re[skip], sa.c_im[skip]);

    let mut d_re = (ar * d0_re - ai * d0_im)
        + (br * d02_re - bi * d02_im)
        + (cr * d03_re - ci * d03_im);
    let mut d_im = (ar * d0_im + ai * d0_re)
        + (br * d02_im + bi * d02_re)
        + (cr * d03_im + ci * d03_re);

    // Check if already escaped at skip point
    if skip < orbit_len {
        let total_re = orbit.z_re[skip] + d_re;
        let total_im = orbit.z_im[skip] + d_im;
        let mag2 = total_re * total_re + total_im * total_im;
        if mag2 > 256.0 {
            return (skip as u32, mag2 as f32);
        }
    }

    // Continue perturbation from skip point
    for n in skip..orbit_len {
        let z_re = orbit.z_re[n];
        let z_im = orbit.z_im[n];

        let new_d_re =
            2.0 * (z_re * d_re - z_im * d_im) + (d_re * d_re - d_im * d_im) + d0_re;
        let new_d_im =
            2.0 * (z_re * d_im + z_im * d_re) + 2.0 * d_re * d_im + d0_im;

        d_re = new_d_re;
        d_im = new_d_im;

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
    }

    (max_iter, 0.0)
}

/// Original perturbation without SA (used when skip == 0).
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

        let new_d_re =
            2.0 * (z_re * d_re - z_im * d_im) + (d_re * d_re - d_im * d_im) + d0_re;
        let new_d_im =
            2.0 * (z_re * d_im + z_im * d_re) + 2.0 * d_re * d_im + d0_im;

        d_re = new_d_re;
        d_im = new_d_im;

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
    }

    (max_iter, 0.0)
}
