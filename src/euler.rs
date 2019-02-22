use std::marker::PhantomData;

use nalgebra::{Real, Vector};

use crate::{Explicit, FixStepTimeEvolution, ModelSpec, VectorStorage};

pub struct Euler<'a, E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    e: &'a E,
    f: Vector<E::Scalar, E::Dim, S>,

    _phantom: PhantomData<R>,
}

impl<'a, E, S, R> Euler<'a, E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
{
    pub fn new(e: &'a E, f: Vector<E::Scalar, E::Dim, S>) -> Euler<'a, E, S, R> {
        Euler {
            e,
            f,
            _phantom: PhantomData,
        }
    }
}

impl<'a, E, S, R> ModelSpec for Euler<'a, E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
    R: Real,
{
    type Scalar = E::Scalar;
    type Dim = E::Dim;
}

impl<'a, E, S, R> FixStepTimeEvolution<S> for Euler<'a, E, S, R>
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
        self.e.force(x, v, &mut self.f);
        x.iter_mut()
            .zip(v.iter_mut())
            .zip(self.f.iter())
            .map(|((x, v), f)| {
                *x += *v * dt.into();
                *v += *f * dt.into();
            })
            .last();
    }
}
