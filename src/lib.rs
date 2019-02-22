pub mod euler;

use nalgebra::{
    storage::{Storage, StorageMut},
    DimName, Real, Scalar, Vector,
};

pub use euler::Euler;

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

pub trait FixStepTimeEvolution<S>: ModelSpec
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
