use rand::{rngs::ThreadRng, Rng};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        if rows * cols != data.len() {
            panic!("Data size doesn't match dimensions");
        }

        Self { rows, cols, data }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn random(rows: usize, cols: usize, min: f64, max: f64) -> Self {
        let mut rng: ThreadRng = rand::rng();
        let mut data: Vec<f64> = Vec::new();

        for _ in 0..(rows * cols) {
            data.push(rng.random_range(min..=max));
        }

        Self { rows, cols, data }
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        self.data[self.cols * row + col]
    }

    pub fn set(&mut self, row: usize, col: usize, value: f64) {
        self.data[self.cols * row + col] = value
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                let n: f64 = self.data[row * self.cols + col];
                write!(f, "{:>12.4} ", n)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
