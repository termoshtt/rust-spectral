
extern crate itertools;
extern crate ndarray;
extern crate ndarray_odeint;
extern crate fftw;
extern crate num_traits;

use itertools::iterate;
use num_traits::Zero;
use ndarray::*;
use ndarray_odeint::prelude::*;
use fftw::*;

struct Mul2 {}

impl StiffDiag<c64, Ix1> for Mul2 {
    fn nonlinear(&self, _: RcArray1<c64>) -> RcArray1<c64> {
        rcarr1(&[c64::zero()])
    }
    fn linear_diagonal(&self) -> RcArray1<c64> {
        rcarr1(&[c64::new(0.0, 1.0)])
    }
}

fn main() {
    let eom = Mul2 {};
    let teo = semi_implicit::diag_rk4(eom, 0.01);
    let ts = iterate(rcarr1(&[c64::new(1.0, 0.0)]), |y| teo.iterate(y.clone()));
    let end_time = 1000;
    for v in ts.take(end_time) {
        println!("{}, {}", v[0].re(), v[0].im());
    }
}
