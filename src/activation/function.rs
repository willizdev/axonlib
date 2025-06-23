use crate::matrix::Matrix;

pub struct ActivationFn {
    output: Option<Matrix>,
}

impl ActivationFn {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn get_output(&self) -> &Option<Matrix> {
        &self.output
    }

    pub fn set_output(&mut self, output: Matrix) {
        self.output = Some(output);
    }
}
