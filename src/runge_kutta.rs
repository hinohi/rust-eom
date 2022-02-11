use num_traits::{One, Zero};

use crate::{Explicit, ModelSpec, TimeEvolution};

pub struct RK1<E>
where
    E: Explicit,
{
    eom: E,
}

impl<E> RK1<E>
where
    E: Explicit,
{
    pub fn new(eom: E) -> RK1<E> {
        RK1 { eom }
    }
}

impl<E> ModelSpec for RK1<E>
where
    E: Explicit,
{
    type Scalar = E::Scalar;
}

impl<E> TimeEvolution for RK1<E>
where
    E: Explicit,
{
    fn iterate(
        &mut self,
        t: &mut Self::Scalar,
        x: &mut [Self::Scalar],
        v: &mut [Self::Scalar],
        dt: Self::Scalar,
    ) -> Self::Scalar {
        let mut a = vec![Self::Scalar::zero(); x.len()];
        self.eom.acceleration(*t, x, v, &mut a);
        x.iter_mut()
            .zip(v.iter_mut())
            .zip(a.iter())
            .map(|((x, v), f)| {
                *x += *v * dt;
                *v += *f * dt;
            })
            .last();
        *t += dt;
        dt
    }
}

pub struct RK2<E>
where
    E: Explicit,
{
    eom: E,
}

impl<E> RK2<E>
where
    E: Explicit,
{
    pub fn new(eom: E) -> RK2<E> {
        RK2 { eom }
    }
}

impl<E> ModelSpec for RK2<E>
where
    E: Explicit,
{
    type Scalar = E::Scalar;
}

impl<E> TimeEvolution for RK2<E>
where
    E: Explicit,
{
    fn iterate(
        &mut self,
        t: &mut Self::Scalar,
        x: &mut [Self::Scalar],
        v: &mut [Self::Scalar],
        dt: Self::Scalar,
    ) -> Self::Scalar {
        let n = x.len();
        assert_eq!(n, v.len());
        let dt2 = dt / (Self::Scalar::one() + Self::Scalar::one());
        // k1
        let mut x1 = vec![Self::Scalar::zero(); n];
        let mut v1 = vec![Self::Scalar::zero(); n];
        let mut a = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t, x, v, &mut a);
        for ((xx, x), v) in x1.iter_mut().zip(x.iter()).zip(v.iter()) {
            *xx = *x + *v * dt2
        }
        for ((vv, v), a) in v1.iter_mut().zip(v.iter()).zip(a.iter()) {
            *vv = *v + *a * dt2
        }
        // k2
        self.eom.acceleration(*t + dt2, &x1, &v1, &mut a);
        // sum
        for (x, &v) in x.iter_mut().zip(v1.iter()) {
            *x += v * dt
        }
        for (v, &a) in v.iter_mut().zip(a.iter()) {
            *v += a * dt
        }
        *t += dt;
        dt
    }
}

pub struct RK4<E>
where
    E: Explicit,
{
    eom: E,
}

impl<E> RK4<E>
where
    E: Explicit,
{
    pub fn new(eom: E) -> RK4<E> {
        RK4 { eom }
    }
}

impl<E> ModelSpec for RK4<E>
where
    E: Explicit,
{
    type Scalar = E::Scalar;
}

impl<E> TimeEvolution for RK4<E>
where
    E: Explicit,
{
    fn iterate(
        &mut self,
        t: &mut Self::Scalar,
        x: &mut [Self::Scalar],
        v: &mut [Self::Scalar],
        dt: Self::Scalar,
    ) -> Self::Scalar {
        let n = x.len();
        assert_eq!(n, v.len());
        let one = Self::Scalar::one();
        let two = one + one;
        let six = two + two + two;
        let dt2 = dt / two;
        let dt6 = dt / six;
        // k1
        let mut x1 = vec![Self::Scalar::zero(); n];
        let mut v1 = vec![Self::Scalar::zero(); n];
        let mut a1 = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t, x, v, &mut a1);
        x1.iter_mut()
            .zip(x.iter())
            .zip(v.iter())
            .map(|((xx, x), v)| *xx = *x + *v * dt2)
            .last();
        v1.iter_mut()
            .zip(v.iter())
            .zip(a1.iter())
            .map(|((vv, v), a)| *vv = *v + *a * dt2)
            .last();
        // k2
        let mut x2 = vec![Self::Scalar::zero(); n];
        let mut v2 = vec![Self::Scalar::zero(); n];
        let mut a2 = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t + dt2, &x1, &v1, &mut a2);
        x2.iter_mut()
            .zip(x.iter())
            .zip(v1.iter())
            .map(|((xx, x), v)| *xx = *x + *v * dt2)
            .last();
        v2.iter_mut()
            .zip(v.iter())
            .zip(a2.iter())
            .map(|((vv, v), a)| *vv = *v + *a * dt2)
            .last();
        // k3
        let mut x3 = vec![Self::Scalar::zero(); n];
        let mut v3 = vec![Self::Scalar::zero(); n];
        let mut a3 = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t + dt2, &x2, &v2, &mut a3);
        x3.iter_mut()
            .zip(x.iter())
            .zip(v2.iter())
            .map(|((xx, x), v)| *xx = *x + *v * dt)
            .last();
        v3.iter_mut()
            .zip(v.iter())
            .zip(a3.iter())
            .map(|((vv, v), a)| *vv = *v + *a * dt)
            .last();
        // k4
        let mut a4 = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t + dt, &x3, &v3, &mut a4);
        // sum
        x.iter_mut()
            .zip(v.iter())
            .zip(v1.iter())
            .zip(v2.iter())
            .zip(v3.iter())
            .map(|((((x, &v), &v1), &v2), &v3)| {
                *x += (v + (v1 + v2) * two + v3) * dt6;
            })
            .last();
        v.iter_mut()
            .zip(a1.iter())
            .zip(a2.iter())
            .zip(a3.iter())
            .zip(a4.iter())
            .map(|((((v, &a1), &a2), &a3), &a4)| {
                *v += (a1 + (a2 + a3) * two + a4) * dt6;
            })
            .last();
        *t += dt;
        dt
    }
}
