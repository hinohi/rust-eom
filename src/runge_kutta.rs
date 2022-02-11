use num_traits::{One, Zero};

use crate::{zip_apply, Explicit, ModelSpec, TimeEvolution};

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
        zip_apply!(
            x in x.iter_mut(), v in v, a in a;
            {
                *x += *v * dt;
                *v += a * dt;
            }
        );
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
        zip_apply!(xx in x1.iter_mut(), &x in x.iter(), &v in v.iter(); *xx = x + v * dt2);
        zip_apply!(vv in v1.iter_mut(), &v in v.iter(), &a in a.iter(); *vv = v + a * dt2);
        // k2
        self.eom.acceleration(*t + dt2, &x1, &v1, &mut a);
        // sum
        zip_apply!(x in x.iter_mut(), &v in v1.iter(); *x += v * dt);
        zip_apply!(v in v.iter_mut(), &a in a.iter(); *v += a * dt);
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
        zip_apply!(xx in x1.iter_mut(), &x in x.iter(), &v in v.iter(); *xx = x + v * dt2);
        zip_apply!(vv in v1.iter_mut(), &v in v.iter(), &a in a1.iter(); *vv = v + a * dt2);
        // k2
        let mut x2 = vec![Self::Scalar::zero(); n];
        let mut v2 = vec![Self::Scalar::zero(); n];
        let mut a2 = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t + dt2, &x1, &v1, &mut a2);
        zip_apply!(xx in x2.iter_mut(), &x in x.iter(), &v in v1.iter(); *xx = x + v * dt2);
        zip_apply!(vv in v2.iter_mut(), &v in v.iter(), &a in a2.iter(); *vv = v + a * dt2);
        // k3
        let mut x3 = vec![Self::Scalar::zero(); n];
        let mut v3 = vec![Self::Scalar::zero(); n];
        let mut a3 = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t + dt2, &x2, &v2, &mut a3);
        zip_apply!(xx in x3.iter_mut(), &x in x.iter(), &v in v2.iter(); *xx = x + v * dt);
        zip_apply!(vv in v3.iter_mut(), &v in v.iter(), &a in a3.iter(); *vv = v + a * dt);
        // k4
        let mut a4 = vec![Self::Scalar::zero(); n];
        self.eom.acceleration(*t + dt, &x3, &v3, &mut a4);
        // sum
        zip_apply!(
            x in x.iter_mut(), &v in v.iter(), &v1 in v1.iter(), &v2 in v2.iter(), &v3 in v3.iter();
            *x += (v + (v1 + v2) * two + v3) * dt6
        );
        zip_apply!(
            v in v.iter_mut(), &a1 in a1.iter(), &a2 in a2.iter(), &a3 in a3.iter(), &a4 in a4.iter();
            *v += (a1 + (a2 + a3) * two + a4) * dt6
        );
        *t += dt;
        dt
    }
}
