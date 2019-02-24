use std::cell::RefCell;
use std::rc::Rc;

use eom_sim::*;
use nalgebra::Vector2;

mod common;
use common::harmonic_oscillator::Oscillator;

#[test]
fn oscillator_rk1() {
    let eom = Rc::new(RefCell::new(Oscillator::new()));
    let mut rk = RK1::new(eom.clone(), Vector2::zeros());
    let mut err = Vec::new();
    for i in 8..16 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.borrow().exact_state(0.0);
        let dt = 2.0_f64.powi(-i);
        for _ in 0..1 << i {
            rk.iterate(&mut t, &mut x, &mut v, dt);
        }
        assert_eq!(t, 1.0);
        let (ex, ev) = eom.borrow().exact_state(t);
        err.push((x - ex).norm() + (v - ev).norm());
    }
    let sig = 1.01;
    for i in 0..err.len() - 1 {
        let rate = err[i] / err[i + 1];
        assert!(2.0 / sig < rate && rate < 2.0 * sig);
    }
}

#[test]
fn oscillator_rk2() {
    let eom = Rc::new(RefCell::new(Oscillator::new()));
    let mut rk = RK2::new(eom.clone(), Vector2::zeros());
    let mut err = Vec::new();
    for i in 8..16 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.borrow().exact_state(0.0);
        let dt = 2.0_f64.powi(-i);
        for _ in 0..1 << i {
            rk.iterate(&mut t, &mut x, &mut v, dt);
        }
        assert_eq!(t, 1.0);
        let (ex, ev) = eom.borrow().exact_state(t);
        err.push((x - ex).norm() + (v - ev).norm());
    }
    let sig = 1.01;
    for i in 0..err.len() - 1 {
        let rate = err[i] / err[i + 1];
        assert!(4.0 / sig < rate && rate < 4.0 * sig);
    }
}

#[test]
fn oscillator_rk4() {
    let eom = Rc::new(RefCell::new(Oscillator::new()));
    let mut rk = RK4::new(eom.clone(), Vector2::zeros());
    let mut err = Vec::new();
    for i in 2..10 {
        let mut t = 0.0;
        let (mut x, mut v) = eom.borrow().exact_state(0.0);
        let dt = 2.0_f64.powi(-i);
        for _ in 0..1 << i {
            rk.iterate(&mut t, &mut x, &mut v, dt);
        }
        assert_eq!(t, 1.0);
        let (ex, ev) = eom.borrow().exact_state(t);
        err.push((x - ex).norm() + (v - ev).norm());
    }
    let sig = 1.01;
    for i in 0..err.len() - 1 {
        let rate = err[i] / err[i + 1];
        println!("{} {} {}", err[i], err[i + 1], rate);
        assert!(16.0 / sig < rate && rate < 16.0 * sig);
    }
}
