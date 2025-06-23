use crate::{activation::function::ActivationFn, matrix::Matrix};

impl ActivationFn {
    pub fn softmax(&mut self, inputs: &Matrix) {
        let mut out: Matrix = inputs.clone();

        for i in 0..out.rows {
            let mut m: f64 = out.get(i, 0);
            for j in 1..out.cols {
                m = m.max(out.get(i, j));
            }
            let mut sum: f64 = 0.0;
            for j in 0..out.cols {
                let exp: f64 = (out.get(i, j) - m).exp();
                out.set(i, j, exp);
                sum += exp;
            }
            for j in 0..out.cols {
                let norm: f64 = out.get(i, j) / sum;
                out.set(i, j, norm);
            }
        }

        self.set_output(out);
    }
}
