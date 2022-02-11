use num_traits::{Num, NumAssign};

pub trait ModelSpec {
    type Scalar: Copy + Num + NumAssign;
}

pub trait Explicit: ModelSpec {
    fn acceleration(
        &self,
        t: Self::Scalar,
        x: &[Self::Scalar],
        v: &[Self::Scalar],
        a: &mut [Self::Scalar],
    );
}

pub trait TimeEvolution: ModelSpec {
    fn iterate(
        &mut self,
        t: &mut Self::Scalar,
        x: &mut [Self::Scalar],
        v: &mut [Self::Scalar],
        dt: Self::Scalar,
    ) -> Self::Scalar;

    fn iterate_n(
        &mut self,
        t: &mut Self::Scalar,
        x: &mut [Self::Scalar],
        v: &mut [Self::Scalar],
        dt: Self::Scalar,
        n: usize,
    ) -> Self::Scalar {
        let mut dt = dt;
        for _ in 0..n {
            dt = self.iterate(t, x, v, dt);
        }
        dt
    }
}
