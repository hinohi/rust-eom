use num_traits::{Num, NumAssign, Zero};

pub trait ModelSpec {
    type Scalar: Copy + Num + NumAssign + PartialOrd;
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

pub trait TimeEvolution<E: Explicit> {
    fn iterate(
        &mut self,
        eom: &E,
        t: &mut E::Scalar,
        x: &mut [E::Scalar],
        v: &mut [E::Scalar],
        a: &[E::Scalar],
        dt: E::Scalar,
    );

    fn iterate_until(
        &mut self,
        eom: &E,
        t: &mut E::Scalar,
        x: &mut [E::Scalar],
        v: &mut [E::Scalar],
        dt: E::Scalar,
        until: E::Scalar,
    ) -> E::Scalar {
        let n = x.len();
        assert_eq!(n, v.len());
        let mut a = vec![E::Scalar::zero(); n];
        while *t < until {
            eom.acceleration(*t, x, v, &mut a);
            self.iterate(eom, t, x, v, &mut a, dt);
        }
        dt
    }
}
