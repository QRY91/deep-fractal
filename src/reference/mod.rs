use rug::Float;

pub struct ReferenceOrbit {
    pub z_re: Vec<f64>,
    pub z_im: Vec<f64>,
    pub escape_iter: u32,
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
