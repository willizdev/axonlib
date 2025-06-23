use crate::{activation::function::ActivationFn, matrix::Matrix};

impl ActivationFn {
    pub fn step(&mut self, inputs: &Matrix) {
        let mut out: Matrix = inputs.clone();

        for i in 0..out.rows {
            for j in 0..out.cols {
                if out.get(i, j) <= 0.0 {
                    out.set(i, j, 0.0);
                } else {
                    out.set(i, j, 1.0);
                }
            }
        }

        self.set_output(out);
    }
}
