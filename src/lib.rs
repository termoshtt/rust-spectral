
extern crate itertools;
extern crate ndarray;
extern crate ndarray_odeint;
extern crate fftw;
extern crate num_traits;
extern crate num_complex;

pub mod kse;

use ndarray::RcArray1;
use fftw::*;

pub fn coef_1d<F>(n: usize, l: f64, f: F) -> RcArray1<c64>
    where F: Fn(f64) -> f64
{
    let mut pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
    for (i, val) in pair.field.iter_mut().enumerate() {
        let x = l * i as f64 / n as f64;
        *val = f(x) / n as f64;
    }
    pair.forward();
    RcArray1::from_iter(pair.coef.iter().cloned())
}
