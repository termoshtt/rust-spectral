
extern crate itertools;
extern crate ndarray;
extern crate ndarray_odeint;
extern crate fftw;
extern crate num_traits;
extern crate easy_storage;

use itertools::iterate;
use ndarray::*;
use ndarray_odeint::prelude::*;
use fftw::*;
use std::f64::consts::PI;
use easy_storage::msgpack::*;
use easy_storage::traits::*;

struct KSE {
    n_field: usize,
    n_coef: usize,
    length: f64,
    u_pair: Pair<f64, c64>,
    ux_pair: Pair<f64, c64>,
}

impl KSE {
    fn new(n: usize, length: f64) -> Self {
        let u_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        let ux_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        KSE {
            n_field: u_pair.field.len(),
            n_coef: u_pair.coef.len(),
            length: length,
            u_pair: u_pair,
            ux_pair: ux_pair,
        }
    }
}

impl StiffDiag<c64, Ix1> for KSE {
    fn nonlinear(&self, _: RcArray1<c64>) -> RcArray1<c64> {
        RcArray::zeros(self.n_coef)
    }
    fn linear_diagonal(&self) -> RcArray1<c64> {
        RcArray::from_iter((0..self.n_coef)
            .map(|i| c64::new(-2.0 * PI * i as f64 / self.length, 0.0)))
    }
}

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

    let st = MsgpackDir::new("data_dir");
    for (t, v) in ts.take(end_time).enumerate() {
        let filename = format!("v{:05}.msg", t);
        st.save_as(&v, &filename).unwrap();
    }
}
