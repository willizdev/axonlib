use crate::{activation::function::ActivationFn, matrix::Matrix};

impl ActivationFn {
    pub fn linear(&mut self, inputs: &Matrix) {
        let out: Matrix = inputs.clone();
        self.set_output(out);
    }
}
