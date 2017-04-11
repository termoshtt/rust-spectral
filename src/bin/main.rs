
extern crate itertools;
extern crate ndarray;
extern crate ndarray_odeint;
extern crate num_complex;
extern crate spectral;
extern crate fftw;

use itertools::iterate;
use num_complex::Complex64 as c64;
use ndarray::*;
use ndarray_odeint::prelude::*;
use std::f64::consts::PI;
use spectral::kse::*;
use fftw::*;

fn init_data(n: usize, l: f64) -> RcArray1<c64> {
    let mut pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
    let k0 = 2.0 * PI / l;
    for (i, val) in pair.field.iter_mut().enumerate() {
        let x = l * i as f64 / n as f64;
        *val = (k0 * x).sin();
    }
    pair.forward();
    RcArray::from_iter(pair.coef.iter().cloned())
}

fn main() {
    let n = 128;
    let l = 12.0;
    let dt = 0.01;
    let eom = KSE::new(n, l);
    let teo = semi_implicit::diag_rk4(eom, dt);

    let x0 = init_data(n, l);
    let ts = iterate(x0, |y| teo.iterate(y.clone()));
    let end_time = 1000;

    print!("time");
    for i in 0..n {
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
