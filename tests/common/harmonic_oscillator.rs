use eom_sim::*;

/// 単振動+減衰+強制振動
///
/// 運動方程式は
///
/// ```math
/// a + 2 \zeta \omega_0 v + \omega_0^2 x = f \cos \omega t
/// ```
#[derive(Clone)]
pub struct Oscillator {
    x0: [f64; 6],
    v0: [f64; 6],
    zeta: [f64; 6],
    omega0: [f64; 6],
    omega: [f64; 6],
    f: [f64; 6],
}

impl Oscillator {
    pub fn new() -> Oscillator {
        Oscillator {
            x0: [1.0, 2.0, 1.5, -1.0, -0.5, 0.0],
            v0: [0.0, 1.0, -2.0, 0.5, 1.5, 0.0],
            zeta: [0.0, 0.5, 1.0, 0.8, 1.5, 0.1],
            omega0: [0.25, 0.5, 1.0, 0.75, 1.25, 0.25],
            omega: [0.0, 1.0, 0.25, 0.8, 0.5, 0.5],
            f: [0.0, 0.1, 0.5, 1.0, -1.0, -0.25],
        }
    }

    pub fn init_state(&self) -> ([f64; 6], [f64; 6]) {
        (self.x0.clone(), self.v0.clone())
    }

    pub fn exact_position(&self, t: f64) -> [f64; 6] {
        let mut position = [0.0; 6];
        for (i, pos) in position.iter_mut().enumerate() {
            let z = self.zeta[i];
            let w0 = self.omega0[i];
            let w = self.omega[i];
            let f = self.f[i];
            let (px0, pv0) = particular_solution(0.0, z, w0, w, f);
            let g = general_solution(t, self.x0[i] - px0, self.v0[i] - pv0, z, w0);
            let p = particular_solution(t, z, w0, w, f).0;
            *pos = g + p;
        }
        position
    }
}

fn general_solution(t: f64, x0: f64, v0: f64, zeta: f64, omega0: f64) -> f64 {
    assert!(zeta >= 0.0);
    if zeta == 0.0 {
        let t = omega0 * t;
        x0 * t.cos() + v0 / omega0 * t.sin()
    } else if zeta < 1.0 {
        let tt = omega0 * (1.0 - zeta * zeta).sqrt();
        let a = (v0 + zeta * omega0 * x0) / tt;
        (-zeta * omega0 * t).exp() * (x0 * (tt * t).cos() + a * (tt * t).sin())
    } else if zeta == 1.0 {
        (-omega0 * t).exp() * (x0 + (v0 + omega0 * x0) * t)
    } else {
        let tt = omega0 * (zeta * zeta - 1.0).sqrt();
        let a = (v0 + zeta * omega0 * x0) / tt;
        (-zeta * omega0 * t).exp() * (x0 * (tt * t).cosh() + a * (tt * t).sinh())
    }
}

fn particular_solution(t: f64, zeta: f64, omega0: f64, omega: f64, f: f64) -> (f64, f64) {
    let t = omega * t;
    let p = 2.0 * zeta * omega0 * omega;
    let q = omega0 * omega0 - omega * omega;
    (
        f * (p * t.sin() + q * t.cos()) / (p * p + q * q),
        f * omega * (p * t.cos() - q * t.sin()) / (p * p + q * q),
    )
}

impl ModelSpec for Oscillator {
    type Scalar = f64;
}

impl Explicit for Oscillator {
    fn acceleration(&self, t: Self::Scalar, x: &[f64], v: &[f64], a: &mut [f64]) {
        for (i, a) in a.iter_mut().enumerate() {
            let p = -2.0 * self.zeta[i] * self.omega0[i] * v[i];
            let q = -self.omega0[i] * self.omega0[i] * x[i];
            let r = self.f[i] * (self.omega[i] * t).cos();
            *a = p + q + r;
        }
    }
}
