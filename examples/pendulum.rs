use eom_sim::utils::norm;
use eom_sim::{
    runge_kutta::RK4,
    utils::{dot, norm2},
    zip_apply, Eom, Explicit, ModelSpec,
};

#[derive(Clone)]
struct Pendulum {
    g: Vec<f64>,
    length: f64,
    tau: f64,
}

impl Pendulum {
    fn new(g: &[f64], length: f64) -> Pendulum {
        let g_norm = norm(g);
        let tau = (length / g_norm).sqrt();
        Pendulum {
            g: g.iter().map(|&g| g * tau * tau / length).collect(),
            length,
            tau,
        }
    }

    fn revive_dimension(&self, t: f64, x: &[f64], v: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
        (
            t * self.tau,
            x.iter().map(|&x| x * self.length).collect(),
            v.iter().map(|&v| v * self.length / self.tau).collect(),
        )
    }
}

impl ModelSpec for Pendulum {
    type Scalar = f64;
}

impl Eom for Pendulum {
    fn acceleration(
        &self,
        _t: Self::Scalar,
        x: &[Self::Scalar],
        v: &[Self::Scalar],
        a: &mut [Self::Scalar],
    ) {
        let lambda = norm2(v) - dot(x, &self.g);
        zip_apply!(a in a.iter_mut(), &x in x.iter(), &g in self.g.iter(); *a = -lambda * x - g);
    }

    fn correct(&self, _t: Self::Scalar, x: &mut [Self::Scalar], _v: &mut [Self::Scalar]) {
        let length = norm(x);
        zip_apply!(x in x.iter_mut(); *x /= length);
    }
}

fn main() {
    let eom = Pendulum::new(&[0.0, 9.8], 0.5);
    let mut rk = RK4::new();
    let theta = std::f64::consts::FRAC_PI_3;
    let dt = 1.0 / 1024.0;
    let mut t = 0.0;
    let mut x = vec![theta.cos(), theta.sin()];
    let mut v = vec![0.0, 0.0];
    while t <= 10.0 {
        let (d_t, d_x, d_v) = eom.revive_dimension(t, &x, &v);
        println!("{} {} {} {} {}", d_t, d_x[0], d_x[1], d_v[0], d_v[1]);
        let until = t + 1.0 / 128.0;
        rk.iterate_until(&eom, &mut t, &mut x, &mut v, dt, until);
    }
}
