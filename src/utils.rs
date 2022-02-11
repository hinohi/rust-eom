use num_traits::{Float, NumAssign};

pub fn norm<S: Float + NumAssign>(a: &[S]) -> S {
    norm2(a).sqrt()
}

pub fn norm2<S: Copy + NumAssign>(a: &[S]) -> S {
    let mut s = S::zero();
    for &a in a {
        s += a * a;
    }
    s
}

pub fn diff_norm<S: Float + NumAssign>(a: &[S], b: &[S]) -> S {
    diff_norm2(a, b).sqrt()
}

pub fn diff_norm2<S: Copy + NumAssign>(a: &[S], b: &[S]) -> S {
    let mut s = S::zero();
    for (&a, &b) in a.iter().zip(b) {
        s += (a - b) * (a - b);
    }
    s
}

pub fn dot<S: Copy + NumAssign>(a: &[S], b: &[S]) -> S {
    let mut s = S::zero();
    for (&a, &b) in a.iter().zip(b) {
        s += a * b;
    }
    s
}
