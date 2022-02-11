use std::marker::PhantomData;

use num_traits::{One, Zero};

use crate::{zip_apply, Explicit, TimeEvolution};

pub struct RK1<E> {
    model: PhantomData<E>,
}

impl<E> RK1<E> {
    pub fn new() -> RK1<E> {
        RK1 { model: PhantomData }
    }
}

impl<E: Explicit> TimeEvolution<E> for RK1<E> {
    fn iterate(
        &mut self,
        _eom: &E,
        t: &mut E::Scalar,
        x: &mut [E::Scalar],
        v: &mut [E::Scalar],
        a: &[E::Scalar],
        dt: E::Scalar,
    ) {
        zip_apply!(x in x.iter_mut(), v in v.iter_mut(), &a in a.iter(); {
            *x += *v * dt;
            *v += a * dt;
        });
        *t += dt;
    }
}

pub struct RK2<E> {
    model: PhantomData<E>,
}

impl<E> RK2<E> {
    pub fn new() -> RK2<E> {
        RK2 { model: PhantomData }
    }
}

impl<E: Explicit> TimeEvolution<E> for RK2<E> {
    fn iterate(
        &mut self,
        eom: &E,
        t: &mut E::Scalar,
        x: &mut [E::Scalar],
        v: &mut [E::Scalar],
        a: &[E::Scalar],
        dt: E::Scalar,
    ) {
        let n = x.len();
        assert_eq!(n, v.len());
        let dt2 = dt / (E::Scalar::one() + E::Scalar::one());
        // k1
        let mut x1 = vec![E::Scalar::zero(); n];
        let mut v1 = vec![E::Scalar::zero(); n];
        zip_apply!(xx in x1.iter_mut(), &x in x.iter(), &v in v.iter(); *xx = x + v * dt2);
        zip_apply!(vv in v1.iter_mut(), &v in v.iter(), &a in a.iter(); *vv = v + a * dt2);
        // k2
        let mut a2 = vec![E::Scalar::zero(); n];
        eom.acceleration(*t + dt2, &x1, &v1, &mut a2);
        // sum
        zip_apply!(x in x.iter_mut(), &v in v1.iter(); *x += v * dt);
        zip_apply!(v in v.iter_mut(), &a in a2.iter(); *v += a * dt);
        *t += dt;
    }
}

pub struct RK4<E> {
    model: PhantomData<E>,
}

impl<E> RK4<E> {
    pub fn new() -> RK4<E> {
        RK4 { model: PhantomData }
    }
}

impl<E: Explicit> TimeEvolution<E> for RK4<E> {
    fn iterate(
        &mut self,
        eom: &E,
        t: &mut E::Scalar,
        x: &mut [E::Scalar],
        v: &mut [E::Scalar],
        a: &[E::Scalar],
        dt: E::Scalar,
    ) {
        let n = x.len();
        assert_eq!(n, v.len());
        let one = E::Scalar::one();
        let two = one + one;
        let six = two + two + two;
        let dt2 = dt / two;
        let dt6 = dt / six;
        let mut x_tmp = vec![E::Scalar::zero(); n];
        let mut v_tmp = vec![E::Scalar::zero(); n];
        // k1
        zip_apply!(xx in x_tmp.iter_mut(), &x in x.iter(), &v in v.iter(); *xx = x + v * dt2);
        zip_apply!(vv in v_tmp.iter_mut(), &v in v.iter(), &a in a.iter(); *vv = v + a * dt2);
        // k2
        let mut v23 = vec![E::Scalar::zero(); n];
        let mut a2 = vec![E::Scalar::zero(); n];
        eom.acceleration(*t + dt2, &x_tmp, &v_tmp, &mut a2);
        zip_apply!(xx in x_tmp.iter_mut(), &x in x.iter(), &v in v_tmp.iter(); *xx = x + v * dt2);
        zip_apply!(vv in v23.iter_mut(), &v in v.iter(), &a in a2.iter(); *vv = v + a * dt2);
        // k3
        let mut a3 = vec![E::Scalar::zero(); n];
        eom.acceleration(*t + dt2, &x_tmp, &v23, &mut a3);
        zip_apply!(xx in x_tmp.iter_mut(), &x in x.iter(), &v in v23.iter(), vv in v_tmp.iter_mut(); {
            *xx = x + v * dt;
            *vv += v;
        });
        zip_apply!(vv in v23.iter_mut(), &v in v.iter(), &a in a3.iter(); *vv = v + a * dt);
        // k4
        let mut a4 = vec![E::Scalar::zero(); n];
        eom.acceleration(*t + dt, &x_tmp, &v23, &mut a4);
        // sum
        zip_apply!(
            x in x.iter_mut(), &v in v.iter(), &v12 in v_tmp.iter(), &v3 in v23.iter();
            *x += (v + v12 * two + v3) * dt6
        );
        zip_apply!(
            v in v.iter_mut(), &a1 in a.iter(), &a2 in a2.iter(), &a3 in a3.iter(), &a4 in a4.iter();
            *v += (a1 + (a2 + a3) * two + a4) * dt6
        );
        *t += dt;
    }
}
