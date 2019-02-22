use eom_sim::*;
use nalgebra::{Vector, Vector1, U1};

struct Oscillator {
    k: f64,
}

impl Oscillator {
    fn new(k: f64) -> Oscillator {
        Oscillator { k }
    }
}

impl ModelSpec for Oscillator {
    type Scalar = f64;
    type Dim = U1;
}

impl<S> Explicit<S> for Oscillator
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    fn force(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, S>,
        _v: &Vector<Self::Scalar, Self::Dim, S>,
        f: &mut Vector<Self::Scalar, Self::Dim, S>,
    ) {
        f[0] = -self.k * x[0];
    }
}

fn main() {
    let eom = Oscillator::new(0.5);
    let mut toe = Euler::new(&eom, Vector1::new(0.0));
    let mut x = Vector1::new(1.0);
    let mut v = Vector1::new(0.0);
    let mut t = 0.0;
    let dt = 0.25;
    for _ in 0..100 {
        toe.evaluate(&mut x, &mut v, dt);
        t += dt;
        println!("{} {} {}", t, x[0], v[0]);
    }
}
