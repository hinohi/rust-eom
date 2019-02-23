use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use nalgebra::{Real, Vector};
use num_traits::FromPrimitive;

use crate::{Explicit, FixedStepTimeEvolution, ModelSpec, VectorStorage};

pub struct Euler<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    eom: Rc<RefCell<E>>,
    a: Vector<E::Scalar, E::Dim, S>,

    _phantom: PhantomData<R>,
}

impl<E, S, R> Euler<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    pub fn new(eom: Rc<RefCell<E>>, a: Vector<E::Scalar, E::Dim, S>) -> Euler<E, S, R> {
        Euler {
            eom,
            a,
            _phantom: PhantomData,
        }
    }
}

impl<E, S, R> ModelSpec for Euler<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    type Scalar = E::Scalar;
    type Dim = E::Dim;
}

impl<E, S, R> FixedStepTimeEvolution<S> for Euler<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
    R: Real,
    E::Scalar: From<R>,
{
    type Time = R;
    fn exact_dt(
        &mut self,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    ) {
        self.eom.borrow_mut().acceleration(x, v, &mut self.a);
        x.iter_mut()
            .zip(v.iter_mut())
            .zip(self.a.iter())
            .map(|((x, v), f)| {
                *x += *v * dt.into();
                *v += *f * dt.into();
            })
            .last();
    }
}

pub struct RK4<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    eom: Rc<RefCell<E>>,
    x1: Vector<E::Scalar, E::Dim, S>,
    x2: Vector<E::Scalar, E::Dim, S>,
    x3: Vector<E::Scalar, E::Dim, S>,
    v1: Vector<E::Scalar, E::Dim, S>,
    v2: Vector<E::Scalar, E::Dim, S>,
    v3: Vector<E::Scalar, E::Dim, S>,
    a1: Vector<E::Scalar, E::Dim, S>,
    a2: Vector<E::Scalar, E::Dim, S>,
    a3: Vector<E::Scalar, E::Dim, S>,
    a4: Vector<E::Scalar, E::Dim, S>,

    _phantom: PhantomData<R>,
}

impl<E, S, R> RK4<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
    Vector<E::Scalar, E::Dim, S>: Clone,
{
    pub fn new(eom: Rc<RefCell<E>>, x: Vector<E::Scalar, E::Dim, S>) -> RK4<E, S, R> {
        RK4 {
            eom,
            x1: x.clone(),
            x2: x.clone(),
            x3: x.clone(),
            v1: x.clone(),
            v2: x.clone(),
            v3: x.clone(),
            a1: x.clone(),
            a2: x.clone(),
            a3: x.clone(),
            a4: x,
            _phantom: PhantomData,
        }
    }
}
impl<E, S, R> ModelSpec for RK4<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    type Scalar = E::Scalar;
    type Dim = E::Dim;
}

impl<E, S, R> FixedStepTimeEvolution<S> for RK4<E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
    R: Real,
    E::Scalar: From<R>,
{
    type Time = R;
    fn exact_dt(
        &mut self,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    ) {
        let mut eom = self.eom.borrow_mut();
        let dt = Self::Scalar::from(dt);
        let dt2 = dt * Self::Scalar::from_f64(0.5).unwrap();
        let dt6 = dt / Self::Scalar::from_f64(6.0).unwrap();
        // k1
        eom.acceleration(x, v, &mut self.a1);
        self.x1
            .iter_mut()
            .zip(x.iter())
            .zip(v.iter())
            .map(|((xx, &x), &v)| {
                *xx = x + v * dt2;
            })
            .last();
        self.v1
            .iter_mut()
            .zip(v.iter())
            .zip(self.a1.iter())
            .map(|((vv, &v), &a)| {
                *vv = v + a * dt2;
            })
            .last();
        // k2
        eom.acceleration(&self.x1, &self.v1, &mut self.a2);
        self.x2
            .iter_mut()
            .zip(x.iter())
            .zip(self.v1.iter())
            .map(|((xx, &x), &v)| {
                *xx = x + v * dt2;
            })
            .last();
        self.v2
            .iter_mut()
            .zip(v.iter())
            .zip(self.a2.iter())
            .map(|((vv, &v), &a)| {
                *vv = v + a * dt2;
            })
            .last();
        // k3
        eom.acceleration(&self.x2, &self.v2, &mut self.a3);
        self.x3
            .iter_mut()
            .zip(x.iter())
            .zip(self.v2.iter())
            .map(|((xx, &x), &v)| {
                *xx = x + v * dt2;
            })
            .last();
        self.v3
            .iter_mut()
            .zip(v.iter())
            .zip(self.a3.iter())
            .map(|((vv, &v), &a)| {
                *vv = v + a * dt2;
            })
            .last();
        // k4
        eom.acceleration(&self.x3, &self.v3, &mut self.a4);
        // sum
        let two = Self::Scalar::from_f64(2.0).unwrap();
        x.iter_mut()
            .zip(v.iter())
            .zip(self.v1.iter())
            .zip(self.v2.iter())
            .zip(self.v3.iter())
            .map(|((((x, &v), &v1), &v2), &v3)| {
                *x += (v + (v1 + v2) * two + v3) * dt6;
            })
            .last();
        v.iter_mut()
            .zip(self.a1.iter())
            .zip(self.a2.iter())
            .zip(self.a3.iter())
            .zip(self.a4.iter())
            .map(|((((v, &a1), &a2), &a3), &a4)| {
                *v += (a1 + (a2 + a3) * two + a4) * dt6;
            })
            .last();
    }
}
