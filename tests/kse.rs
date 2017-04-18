
extern crate ndarray;
extern crate ndarray_odeint;
extern crate ndarray_numtest;
extern crate fftw;
extern crate spectral;

use ndarray_odeint::traits::*;
use ndarray_numtest::prelude::*;
use spectral::kse::*;
use spectral::coef_1d;
use std::f64::consts::PI;

#[test]
fn kse_rhs() {
    let n = 32;
    let l = 1.2334;
    let k0 = 2.0 * PI / l;
    let mut kse = KSE::new(n, l);
    let k = 3.0 * k0;
    let x = coef_1d(n, l, |x| (k * x).cos());
    let ans = coef_1d(n, l, |x| 0.5 * k * (2.0 * k * x).sin());
    println!("x = {:?}", &x);
    let fx = kse.rhs(x);
    println!("fx = {:?}", &fx);
    println!("ans = {:?}", &ans);
    fx.to_owned().assert_allclose_l2(&ans.to_owned(), 1e-7);
}
