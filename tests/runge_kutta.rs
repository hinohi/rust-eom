use eom_sim::{runge_kutta::*, utils::diff_norm, TimeEvolution};

mod common;
use common::harmonic_oscillator::Oscillator;

#[test]
fn oscillator_rk1() {
    let eom = Oscillator::new();
    let mut rk = RK1::new();
    let mut err = vec![vec![], vec![], vec![]];
    for i in 8..16 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.init_state();
        let dt = 2.0_f64.powi(-i);
        for j in 0..err.len() {
            rk.iterate_until(&eom, &mut t, &mut x, &mut v, dt, (j + 1) as f64);
            assert_eq!(t, (j + 1) as f64);
            let ex = eom.exact_position(t);
            err[j].push(diff_norm(&x, &ex));
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
    let mut rk = RK2::new();
    let mut err = vec![vec![], vec![], vec![]];
    for i in 6..14 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.init_state();
        let dt = 2.0_f64.powi(-i);
        for j in 0..err.len() {
            rk.iterate_until(&eom, &mut t, &mut x, &mut v, dt, (j + 1) as f64);
            assert_eq!(t, (j + 1) as f64);
            let ex = eom.exact_position(t);
            err[j].push(diff_norm(&x, &ex));
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
    let mut rk = RK4::new();
    let mut err = vec![vec![], vec![], vec![]];
    for i in 2..10 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.init_state();
        let dt = 2.0_f64.powi(-i);
        for j in 0..err.len() {
            rk.iterate_until(&eom, &mut t, &mut x, &mut v, dt, (j + 1) as f64);
            assert_eq!(t, (j + 1) as f64);
            let ex = eom.exact_position(t);
            err[j].push(diff_norm(&x, &ex));
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
