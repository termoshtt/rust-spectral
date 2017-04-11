
use fftw::*;
use ndarray::prelude::*;
use ndarray::RcArray1;
use ndarray_odeint::prelude::*;
use std::f64::consts::PI;

pub struct KSE {
    n_coef: usize,
    length: f64,
    u_pair: Pair<f64, c64>,
    ux_pair: Pair<f64, c64>,
}

impl KSE {
    pub fn new(n: usize, length: f64) -> Self {
        let u_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        let ux_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        KSE {
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
