use eom_sim::utils::norm;
use eom_sim::{
    runge_kutta::RK4,
    utils::{dot, norm2},
    zip_apply, Explicit, ModelSpec, TimeEvolution,
};

#[derive(Clone)]
struct Pendulum {
    g: Vec<f64>,
}

impl ModelSpec for Pendulum {
    type Scalar = f64;
}

impl Explicit for Pendulum {
    fn acceleration(
        &self,
        _t: Self::Scalar,
        x: &[Self::Scalar],
        v: &[Self::Scalar],
        a: &mut [Self::Scalar],
    ) {
        let length2 = norm2(x);
        let lambda = (norm2(v) - dot(x, &self.g)) / (2.0 * length2);
        zip_apply!(a in a.iter_mut(), &x in x.iter(), &g in self.g.iter(); *a = -2.0 * lambda * x - g);
    }
}

fn main() {
    let eom = Pendulum { g: vec![0.0, 9.8] };
    let mut rk = RK4::new(eom);
    let length = 0.5;
    let theta = std::f64::consts::FRAC_PI_3;
    let mut dt = 1.0 / 128.0;
    let mut t = 0.0;
    let mut x = vec![length * theta.cos(), length * theta.sin()];
    let mut v = vec![0.0, 0.0];
    while t <= 10.0 {
        println!("{} {} {} {} {}", t, x[0], x[1], v[0], v[1]);
        dt = rk.iterate(&mut t, &mut x, &mut v, dt);
        let l = length / norm(&x);
        zip_apply!(x in x.iter_mut(); *x *= l);
    }
}
