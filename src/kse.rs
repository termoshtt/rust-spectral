
use fftw::*;
use ndarray::prelude::*;
use ndarray::RcArray1;
use ndarray_odeint::prelude::*;
use std::f64::consts::PI;

pub struct KSE {
    n_coef: usize,
    u_pair: Pair<f64, c64>,
    ux_pair: Pair<f64, c64>,
    k: RcArray1<c64>,
}

impl KSE {
    pub fn new(n: usize, length: f64) -> Self {
        let u_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        let ux_pair = Pair::r2c_1d(n, FLAG::FFTW_ESTIMATE);
        let n_coef = u_pair.coef.len();
        KSE {
            n_coef: n_coef,
            u_pair: u_pair,
            ux_pair: ux_pair,
            k: RcArray::from_iter((0..n_coef).map(|i| c64::new(0.0, 2.0 * PI * i as f64 / length))),
        }
    }
}

impl<'a> EOM<c64, Ix1> for &'a mut KSE {
    fn rhs(self, mut u: RcArray1<c64>) -> RcArray1<c64> {
        for i in 0..self.n_coef {
            self.u_pair.coef[i] = u[i];
            self.ux_pair.coef[i] = self.k[i] * u[i];
        }
        self.u_pair.backward();
        self.ux_pair.backward();
        for (up, uxp) in self.u_pair.field.iter_mut().zip(self.ux_pair.field.iter()) {
            *up = *up * *uxp;
        }
        self.u_pair.forward();
        for (up, u) in self.u_pair.coef.iter().zip(u.iter_mut()) {
            *u = *up;
        }
        u
    }
}

impl Diag<c64, Ix1> for KSE {
    fn diagonal(&self) -> RcArray1<c64> {
        let k2 = &self.k * &self.k;
        let k4 = &k2 * &k2;
        (-k2 + k4).into_shared()
    }
}
