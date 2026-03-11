use rug::Float;

pub struct ReferenceOrbit {
    pub z_re: Vec<f64>,
    pub z_im: Vec<f64>,
    pub escape_iter: u32,
}

/// Series approximation coefficients: δ_n ≈ A_n·δ₀ + B_n·δ₀² + C_n·δ₀³
pub struct SeriesApprox {
    pub a_re: Vec<f64>,
    pub a_im: Vec<f64>,
    pub b_re: Vec<f64>,
    pub b_im: Vec<f64>,
    pub c_re: Vec<f64>,
    pub c_im: Vec<f64>,
    pub valid: usize,
}

impl SeriesApprox {
    /// Compute SA coefficients from a reference orbit.
    pub fn from_orbit(orbit: &ReferenceOrbit) -> Self {
        let len = orbit.z_re.len();
        let mut a_re = Vec::with_capacity(len + 1);
        let mut a_im = Vec::with_capacity(len + 1);
        let mut b_re = Vec::with_capacity(len + 1);
        let mut b_im = Vec::with_capacity(len + 1);
        let mut c_re = Vec::with_capacity(len + 1);
        let mut c_im = Vec::with_capacity(len + 1);

        // Initial: δ_0 = δ₀, so A_0 = 1, B_0 = 0, C_0 = 0
        let (mut ar, mut ai) = (1.0_f64, 0.0_f64);
        let (mut br, mut bi) = (0.0_f64, 0.0_f64);
        let (mut cr, mut ci) = (0.0_f64, 0.0_f64);

        a_re.push(ar); a_im.push(ai);
        b_re.push(br); b_im.push(bi);
        c_re.push(cr); c_im.push(ci);

        let mut valid = 1;

        for n in 0..len {
            let zr = orbit.z_re[n];
            let zi = orbit.z_im[n];

            // C_{n+1} = 2·Z_n·C_n + 2·A_n·B_n
            let ab_re = ar * br - ai * bi;
            let ab_im = ar * bi + ai * br;
            let new_cr = 2.0 * (zr * cr - zi * ci) + 2.0 * ab_re;
            let new_ci = 2.0 * (zr * ci + zi * cr) + 2.0 * ab_im;

            // B_{n+1} = 2·Z_n·B_n + A_n²
            let a2_re = ar * ar - ai * ai;
            let a2_im = 2.0 * ar * ai;
            let new_br = 2.0 * (zr * br - zi * bi) + a2_re;
            let new_bi = 2.0 * (zr * bi + zi * br) + a2_im;

            // A_{n+1} = 2·Z_n·A_n + 1
            let new_ar = 2.0 * (zr * ar - zi * ai) + 1.0;
            let new_ai = 2.0 * (zr * ai + zi * ar);

            if !new_ar.is_finite() || !new_ai.is_finite()
                || !new_br.is_finite() || !new_bi.is_finite()
                || !new_cr.is_finite() || !new_ci.is_finite()
            {
                break;
            }

            ar = new_ar; ai = new_ai;
            br = new_br; bi = new_bi;
            cr = new_cr; ci = new_ci;

            a_re.push(ar); a_im.push(ai);
            b_re.push(br); b_im.push(bi);
            c_re.push(cr); c_im.push(ci);
            valid = n + 2;
        }

        Self { a_re, a_im, b_re, b_im, c_re, c_im, valid }
    }

    /// Find global skip point for max pixel delta magnitude squared.
    pub fn skip_point(&self, max_d0_mag2: f64) -> usize {
        let tol2 = 1e-3_f64 * 1e-3;
        let d0_mag4 = max_d0_mag2 * max_d0_mag2;

        let mut skip = 0;
        for n in 1..self.valid {
            let a_mag2 = self.a_re[n] * self.a_re[n] + self.a_im[n] * self.a_im[n];
            let c_mag2 = self.c_re[n] * self.c_re[n] + self.c_im[n] * self.c_im[n];

            // |C_n·δ₀²| < tol·|A_n| → |C_n|²·|δ₀|⁴ < tol²·|A_n|²
            if c_mag2 > 0.0 && c_mag2 * d0_mag4 >= tol2 * a_mag2 {
                break;
            }
            skip = n;
        }
        skip
    }
}

impl ReferenceOrbit {
    /// Compute a reference orbit at the given center using arbitrary precision.
    /// Returns the orbit as f64 pairs for GPU/perturbation use.
    pub fn compute(center_re: &Float, center_im: &Float, max_iter: u32, zoom: f64) -> Self {
        // Precision scales with zoom depth: ~3.32 bits per decimal digit + margin
        let digits = (zoom.max(1.0).log10() + 20.0) as u32;
        let prec = (digits as f64 * 3.322) as u32;
        let prec = prec.max(128).min(16000);

        let c_re = Float::with_val(prec, center_re);
        let c_im = Float::with_val(prec, center_im);

        let mut z_re = Float::with_val(prec, 0.0);
        let mut z_im = Float::with_val(prec, 0.0);

        let mut orbit_re = Vec::with_capacity(max_iter as usize);
        let mut orbit_im = Vec::with_capacity(max_iter as usize);

        for i in 0..max_iter {
            orbit_re.push(z_re.to_f64());
            orbit_im.push(z_im.to_f64());

            // Escape check
            let zr2 = Float::with_val(prec, &z_re * &z_re);
            let zi2 = Float::with_val(prec, &z_im * &z_im);
            let mag2 = Float::with_val(prec, &zr2 + &zi2);
            if mag2 > 256.0 {
                return Self {
                    z_re: orbit_re,
                    z_im: orbit_im,
                    escape_iter: i,
                };
            }

            // z = z² + c
            // new_im = 2 * z_re * z_im + c_im
            let mut new_im = Float::with_val(prec, &z_re * &z_im);
            new_im *= 2u32;
            new_im += &c_im;
            // new_re = z_re² - z_im² + c_re
            let mut new_re = Float::with_val(prec, &zr2 - &zi2);
            new_re += &c_re;
            z_re = new_re;
            z_im = new_im;
        }

        Self {
            z_re: orbit_re,
            z_im: orbit_im,
            escape_iter: max_iter,
        }
    }
}
