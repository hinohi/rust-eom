use eom_sim::*;

mod common;
use common::harmonic_oscillator::Oscillator;

fn error_norm(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&a, &b) in a.iter().zip(b) {
        s += (a - b) * (a - b);
    }
    s.sqrt()
}

#[test]
fn oscillator_rk1() {
    let eom = Oscillator::new();
    let mut rk = RK1::new(eom.clone());
    let mut err = vec![vec![], vec![], vec![]];
    for i in 8..16 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.init_state();
        let dt = 2.0_f64.powi(-i);
        for j in 0..err.len() {
            rk.iterate_n(&mut t, &mut x, &mut v, dt, (1 << i) as usize);
            assert_eq!(t, (j + 1) as f64);
            let ex = eom.exact_position(t);
            err[j].push(error_norm(&x, &ex));
        }
    }
    for e in err {
        for i in 0..e.len() - 1 {
            let rate = e[i] / e[i + 1];
            println!("{} {} {}", e[i], e[i + 1], rate);
            assert!(2.0 * 0.9 < rate && rate < 4.0);
        }
    }
}

#[test]
fn oscillator_rk2() {
    let eom = Oscillator::new();
    let mut rk = RK2::new(eom.clone());
    let mut err = vec![vec![], vec![], vec![]];
    for i in 6..14 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.init_state();
        let dt = 2.0_f64.powi(-i);
        for j in 0..err.len() {
            rk.iterate_n(&mut t, &mut x, &mut v, dt, (1 << i) as usize);
            assert_eq!(t, (j + 1) as f64);
            let ex = eom.exact_position(t);
            err[j].push(error_norm(&x, &ex));
        }
    }
    for e in err {
        for i in 0..e.len() - 1 {
            let rate = e[i] / e[i + 1];
            println!("{} {} {}", e[i], e[i + 1], rate);
            assert!(4.0 * 0.9 < rate && rate < 8.0);
        }
    }
}

#[test]
fn oscillator_rk4() {
    let eom = Oscillator::new();
    let mut rk = RK4::new(eom.clone());
    let mut err = vec![vec![], vec![], vec![]];
    for i in 2..10 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.init_state();
        let dt = 2.0_f64.powi(-i);
        for j in 0..err.len() {
            rk.iterate_n(&mut t, &mut x, &mut v, dt, (1 << i) as usize);
            assert_eq!(t, (j + 1) as f64);
            let ex = eom.exact_position(t);
            err[j].push(error_norm(&x, &ex));
        }
    }
    for e in err {
        for i in 0..e.len() - 1 {
            let rate = e[i] / e[i + 1];
            println!("{} {} {}", e[i], e[i + 1], rate);
            assert!(16.0 * 0.9 < rate && rate < 32.0);
        }
    }
}
