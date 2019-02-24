use std::cell::RefCell;
use std::rc::Rc;

use nalgebra::Vector;
use num_traits::cast::FromPrimitive;

use crate::{Explicit, FixedStepTimeEvolution, ModelSpec, VectorStorage};

pub struct RK1<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    eom: Rc<RefCell<E>>,
    a: Vector<E::Scalar, E::Dim, S>,
}

impl<E, S> RK1<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    pub fn new(eom: Rc<RefCell<E>>, a: Vector<E::Scalar, E::Dim, S>) -> RK1<E, S> {
        RK1 { eom, a }
    }
}

impl<E, S> ModelSpec for RK1<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    type Time = E::Time;
    type Scalar = E::Scalar;
    type Dim = E::Dim;
}

impl<E, S> FixedStepTimeEvolution<S> for RK1<E, S>
where
    E: Explicit<S>,
    E::Scalar: From<E::Time>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    fn iterate(
        &mut self,
        t: &mut Self::Time,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    ) {
        let h = Self::Scalar::from(dt);
        let mut eom = self.eom.borrow_mut();
        eom.acceleration(*t, x, v, &mut self.a);
        x.iter_mut()
            .zip(v.iter_mut())
            .zip(self.a.iter())
            .map(|((x, v), f)| {
                *x += *v * h;
                *v += *f * h;
            })
            .last();
        *t += dt;
    }
}

pub struct RK2<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    eom: Rc<RefCell<E>>,
    x1: Vector<E::Scalar, E::Dim, S>,
    v1: Vector<E::Scalar, E::Dim, S>,
    a1: Vector<E::Scalar, E::Dim, S>,
    a2: Vector<E::Scalar, E::Dim, S>,
}

impl<E, S> RK2<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
    Vector<E::Scalar, E::Dim, S>: Clone,
{
    pub fn new(eom: Rc<RefCell<E>>, a: Vector<E::Scalar, E::Dim, S>) -> RK2<E, S> {
        RK2 {
            eom,
            x1: a.clone(),
            v1: a.clone(),
            a1: a.clone(),
            a2: a,
        }
    }
}

impl<E, S> ModelSpec for RK2<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    type Time = E::Time;
    type Scalar = E::Scalar;
    type Dim = E::Dim;
}

impl<E, S> FixedStepTimeEvolution<S> for RK2<E, S>
where
    E: Explicit<S>,
    E::Scalar: From<E::Time>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    fn iterate(
        &mut self,
        t: &mut Self::Time,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    ) {
        let h = Self::Scalar::from(dt);
        let h2 = h * Self::Scalar::from_f64(0.5).unwrap();
        let dt2 = dt * Self::Time::from_f64(0.5).unwrap();
        let mut eom = self.eom.borrow_mut();
        // k1
        eom.acceleration(*t, x, v, &mut self.a1);
        self.x1
            .iter_mut()
            .zip(x.iter())
            .zip(v.iter())
            .map(|((xx, x), v)| *xx = *x + *v * h2)
            .last();
        self.v1
            .iter_mut()
            .zip(v.iter())
            .zip(self.a1.iter())
            .map(|((vv, v), a)| *vv = *v + *a * h2)
            .last();
        // k2
        eom.acceleration(*t + dt2, &self.x1, &self.v1, &mut self.a2);
        // sum
        x.iter_mut()
            .zip(self.v1.iter())
            .map(|(x, &v)| *x += v * h)
            .last();
        v.iter_mut()
            .zip(self.a2.iter())
            .map(|(v, &a)| *v += a * h)
            .last();
        *t += dt;
    }
}

pub struct RK4<E, S>
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
}

impl<E, S> RK4<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
    Vector<E::Scalar, E::Dim, S>: Clone,
{
    pub fn new(eom: Rc<RefCell<E>>, a: Vector<E::Scalar, E::Dim, S>) -> RK4<E, S> {
        RK4 {
            eom,
            x1: a.clone(),
            x2: a.clone(),
            x3: a.clone(),
            v1: a.clone(),
            v2: a.clone(),
            v3: a.clone(),
            a1: a.clone(),
            a2: a.clone(),
            a3: a.clone(),
            a4: a,
        }
    }
}

impl<E, S> ModelSpec for RK4<E, S>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    type Time = E::Time;
    type Scalar = E::Scalar;
    type Dim = E::Dim;
}

impl<E, S> FixedStepTimeEvolution<S> for RK4<E, S>
where
    E: Explicit<S>,
    E::Scalar: From<E::Time>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    fn iterate(
        &mut self,
        t: &mut Self::Time,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    ) {
        let h = Self::Scalar::from(dt);
        let h2 = h * Self::Scalar::from_f64(0.5).unwrap();
        let h6 = h / Self::Scalar::from_f64(6.0).unwrap();
        let dt2 = dt * Self::Time::from_f64(0.5).unwrap();
        let mut eom = self.eom.borrow_mut();
        // k1
        eom.acceleration(*t, x, v, &mut self.a1);
        self.x1
            .iter_mut()
            .zip(x.iter())
            .zip(v.iter())
            .map(|((xx, x), v)| *xx = *x + *v * h2)
            .last();
        self.v1
            .iter_mut()
            .zip(v.iter())
            .zip(self.a1.iter())
            .map(|((vv, v), a)| *vv = *v + *a * h2)
            .last();
        // k2
        eom.acceleration(*t + dt2, &self.x1, &self.v1, &mut self.a2);
        self.x2
            .iter_mut()
            .zip(x.iter())
            .zip(self.v1.iter())
            .map(|((xx, x), v)| *xx = *x + *v * h2)
            .last();
        self.v2
            .iter_mut()
            .zip(v.iter())
            .zip(self.a2.iter())
            .map(|((vv, v), a)| *vv = *v + *a * h2)
            .last();
        // k3
        eom.acceleration(*t + dt2, &self.x2, &self.v2, &mut self.a3);
        self.x3
            .iter_mut()
            .zip(x.iter())
            .zip(self.v2.iter())
            .map(|((xx, x), v)| *xx = *x + *v * h)
            .last();
        self.v3
            .iter_mut()
            .zip(v.iter())
            .zip(self.a3.iter())
            .map(|((vv, v), a)| *vv = *v + *a * h)
            .last();
        // k4
        eom.acceleration(*t + dt, &self.x3, &self.v3, &mut self.a4);
        // sum
        let two = Self::Scalar::from_f64(2.0).unwrap();
        x.iter_mut()
            .zip(v.iter())
            .zip(self.v1.iter())
            .zip(self.v2.iter())
            .zip(self.v3.iter())
            .map(|((((x, &v), &v1), &v2), &v3)| {
                *x += (v + (v1 + v2) * two + v3) * h6;
            })
            .last();
        v.iter_mut()
            .zip(self.a1.iter())
            .zip(self.a2.iter())
            .zip(self.a3.iter())
            .zip(self.a4.iter())
            .map(|((((v, &a1), &a2), &a3), &a4)| {
                *v += (a1 + (a2 + a3) * two + a4) * h6;
            })
            .last();
        *t += dt;
    }
}
