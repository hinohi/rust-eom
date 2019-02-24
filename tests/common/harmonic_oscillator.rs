use nalgebra::{Vector, Vector2, U2};

use eom_sim::*;

pub struct Oscillator;

impl Oscillator {
    pub fn new() -> Oscillator {
        Oscillator {}
    }

    pub fn exact_state(&self, t: f64) -> (Vector2<f64>, Vector2<f64>) {
        (
            Vector2::new(2.0 * t.cos(), 3.0 * (t * 0.5).sin()),
            Vector2::new(-2.0 * t.sin(), 1.5 * (t * 0.5).cos()),
        )
    }
}

impl ModelSpec for Oscillator {
    type Time = f64;
    type Scalar = f64;
    type Dim = U2;
}

impl<S> Explicit<S> for Oscillator
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    fn acceleration(
        &mut self,
        _t: Self::Time,
        x: &Vector<Self::Scalar, Self::Dim, S>,
        _v: &Vector<Self::Scalar, Self::Dim, S>,
        a: &mut Vector<Self::Scalar, Self::Dim, S>,
    ) {
        a[0] = -x[0];
        a[1] = -0.25 * x[1];
    }
}
