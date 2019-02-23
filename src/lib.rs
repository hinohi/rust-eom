pub mod explicit_fixed_step;

use nalgebra::{
    storage::{Storage, StorageMut},
    DimName, Real, Scalar, Vector,
};

pub use explicit_fixed_step::Euler;

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
    fn acceleration(
        &self,
        x: &Vector<Self::Scalar, Self::Dim, S>,
        v: &Vector<Self::Scalar, Self::Dim, S>,
        a: &mut Vector<Self::Scalar, Self::Dim, S>,
    );
}

pub trait FixedStepTimeEvolution<S>: ModelSpec
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    type Time: Real;
    fn exact_dt(
        &mut self,
        x: &mut Vector<Self::Scalar, Self::Dim, S>,
        v: &mut Vector<Self::Scalar, Self::Dim, S>,
        dt: Self::Time,
    );
}
