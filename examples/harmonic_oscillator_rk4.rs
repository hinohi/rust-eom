use eom_sim::explicit_fixed_step::RK4;
use eom_sim::*;
use nalgebra::{Vector, Vector1, U1};

struct Oscillator {
    k: f64,
    m: f64,
}

impl ModelSpec for Oscillator {
    type Scalar = f64;
    type Dim = U1;
}

impl Oscillator {
    fn new(m: f64, k: f64) -> Oscillator {
        Oscillator { k, m }
    }

    fn energy<S>(&self, x: &Vector<f64, U1, S>, v: &Vector<f64, U1, S>) -> f64
    where
        S: VectorStorage<f64, U1>,
    {
        (self.k * x[0] * x[0] + self.m * v[0] * v[0]) * 0.5
    }
}

impl<S> Explicit<S> for Oscillator
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    fn acceleration(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, S>,
        _v: &Vector<Self::Scalar, Self::Dim, S>,
        a: &mut Vector<Self::Scalar, Self::Dim, S>,
    ) {
        a[0] = -self.k * x[0] / self.m;
    }
}

fn main() {
    let eom = Oscillator::new(1.0, 4.0);
    let mut rk4 = RK4::new(&eom, Vector1::new(0.0));
    let mut x = Vector1::new(1.0);
    let mut v = Vector1::new(0.0);
    let mut t = 0.0;
    let dt = 2.0_f64.powi(-10);
    for _ in 0..10000 {
        rk4.exact_dt(&mut x, &mut v, dt);
        t += dt;
        println!("{} {} {} {}", t, x[0], v[0], eom.energy(&x, &v));
    }
}
