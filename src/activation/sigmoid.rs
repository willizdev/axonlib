use std::f64::consts::E;

use crate::{activation::function::ActivationFn, matrix::Matrix};

impl ActivationFn {
    pub fn sigmoid(&mut self, inputs: &Matrix) {
        let mut out: Matrix = inputs.clone();

        for i in 0..out.rows {
            for j in 0..out.cols {
                let n: f64 = out.get(i, j);
                let m: f64 = E.powf(n);
                let k: f64 = m / (1.0 + m);
                out.set(i, j, k);
            }
        }

        self.set_output(out);
    }
}
