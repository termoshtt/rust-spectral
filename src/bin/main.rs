
extern crate itertools;
extern crate ndarray_odeint;
extern crate spectral;

use std::f64::consts::PI;
use itertools::iterate;
use ndarray_odeint::prelude::*;

use spectral::kse::*;
use spectral::coef_1d;

fn main() {
    let n = 128;
    let l = 32.0;
    let k0 = 2.0 * PI / l;
    let dt = 1e-4;
    let eom = KSE::new(n, l);
    let mut teo = semi_implicit::diag_rk4(eom, dt);

    let x0 = coef_1d(n, l, |x| (3.0 * k0 * x).cos() + (2.0 * k0 * x).cos());
    let n_coef = x0.len();
    let ts = iterate(x0, |y| teo.iterate(y.clone()));
    let end_time = 10000;

    print!("time");
    for i in 0..n_coef {
        print!(",r{},c{}", i, i);
    }
    println!("");
    for (t, v) in ts.take(end_time).enumerate() {
        print!("{:e}", dt * t as f64);
        for c in v.iter() {
            print!(",{:e},{:e}", c.re, c.im);
        }
        println!("");
    }
}
