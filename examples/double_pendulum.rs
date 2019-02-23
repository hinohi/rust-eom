use std::cell::RefCell;
use std::rc::Rc;

use eom_sim::*;
use nalgebra::{Matrix2, Vector, Vector2, U2};

struct DoublePendulum {
    g: f64,
    mat: Matrix2<f64>,
}

impl ModelSpec for DoublePendulum {
    type Scalar = f64;
    type Dim = U2;
}

impl DoublePendulum {
    fn new(g: f64) -> DoublePendulum {
        let mut mat = Matrix2::zeros();
        mat[(0, 0)] = 2.0;
        mat[(1, 1)] = 1.0;
        DoublePendulum { g, mat }
    }

    fn energy<S>(&self, x: &Vector<f64, U2, S>, v: &Vector<f64, U2, S>) -> f64
    where
        S: VectorStorage<f64, U2>,
    {
        let t = (2.0 * v[0] * v[0] + v[1] * v[1]) * 0.5 + v[0] * v[1] * (x[0] - x[1]).cos();
        let u = -self.g * (2.0 * x[0].cos() + x[1].cos());
        t + u
    }
}

impl<S> Explicit<S> for DoublePendulum
where
    S: VectorStorage<Self::Scalar, Self::Dim>,
{
    fn acceleration(
        &mut self,
        x: &Vector<Self::Scalar, Self::Dim, S>,
        v: &Vector<Self::Scalar, Self::Dim, S>,
        a: &mut Vector<Self::Scalar, Self::Dim, S>,
    ) {
        let s = (x[1] - x[0]).sin();
        a[0] = s * v[1] * v[1] - self.g * 2.0 * x[0].sin();
        a[1] = -s * v[0] * v[0] - self.g * x[1].sin();
        let c = (x[0] - x[1]).cos();
        self.mat[(0, 1)] = c;
        self.mat[(1, 0)] = c;
        let lu = self.mat.lu();
        lu.solve_mut(a);
    }
}

fn main() {
    let eom = Rc::new(RefCell::new(DoublePendulum::new(1.0)));
    let mut rk = RK4::new(eom.clone(), Vector2::zeros());
    let mut x = Vector2::new(2.0, 2.0);
    let mut v = Vector2::new(0.0, 0.0);
    let mut t = 0.0;
    let dt = 2.0_f64.powi(-10);
    for _ in 0..10000 {
        rk.exact_dt(&mut x, &mut v, dt);
        t += dt;
        println!(
            "{} {} {} {} {} {}",
            t,
            x[0],
            x[1],
            v[0],
            v[1],
            eom.borrow().energy(&x, &v)
        );
    }
}
