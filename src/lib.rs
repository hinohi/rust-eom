use std::marker::PhantomData;

use nalgebra::{
    storage::{Storage, StorageMut},
    DimName, Real, Scalar, Vector,
};

pub trait VectorStorage<N, D>: Storage<N, D> + StorageMut<N, D>
where
    N: Scalar,
    D: DimName,
{
}

impl<N, D, T> VectorStorage<N, D> for T
where
    N: Scalar,
    D: DimName,
    T: Storage<N, D> + StorageMut<N, D>,
{
}

pub trait ModelSpec {
    type Scalar: Scalar + Real;
    type Dim: DimName;
    fn model_size(&self) -> usize {
        Self::Dim::dim()
    }
}

pub trait Explicit<S>: ModelSpec
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    fn force(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, S>,
        v: &Vector<Self::Scalar, Self::Dim, S>,
        f: &mut Vector<Self::Scalar, Self::Dim, S>,
    );
}

pub trait FixTimeStepEvolution<S>: ModelSpec
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    type Time: Real;
    fn evaluate(
        &mut self,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    );
}

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

impl<'a, E, S, R> FixTimeStepEvolution<S> for Euler<'a, E, S, R>
where
    E: Explicit<S>,
    S: VectorStorage<E::Scalar, E::Dim>,
    R: Real,
    E::Scalar: From<R>,
{
    type Time = R;
    fn evaluate(
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
