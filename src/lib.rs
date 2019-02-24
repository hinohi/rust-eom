pub mod runge_kutta;

use nalgebra::{
    storage::{Storage, StorageMut},
    Dim, Real, Scalar, Vector,
};

pub use runge_kutta::*;

pub trait VectorStorage<N, D>: Storage<N, D> + StorageMut<N, D>
where
    N: Scalar,
    D: Dim,
{
}

impl<N, D, T> VectorStorage<N, D> for T
where
    N: Scalar,
    D: Dim,
    T: Storage<N, D> + StorageMut<N, D>,
{
}

pub trait ModelSpec {
    type Time: Real;
    type Scalar: Scalar + Real;
    type Dim: Dim;
}

pub trait Explicit<S>: ModelSpec
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    fn acceleration(
        &mut self,
        t: Self::Time,
        x: &Vector<Self::Scalar, Self::Dim, S>,
        v: &Vector<Self::Scalar, Self::Dim, S>,
        a: &mut Vector<Self::Scalar, Self::Dim, S>,
    );
}

pub trait TimeEvolution<S>: ModelSpec
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    fn iterate(
        &mut self,
        t: &mut Self::Time,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    );

    fn iterate_n(
        &mut self,
        t: &mut Self::Time,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
        n: usize,
    ) {
        for _ in 0..n {
            self.iterate(t, x, v, dt);
        }
    }
}
