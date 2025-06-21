use crate::matrix::Matrix;

pub struct Math;

impl Math {
    pub fn product(a: &Matrix, b: &Matrix) -> Matrix {
        if a.cols != b.rows {
            panic!("Matrix size doesn't match");
        }

        let mut res: Vec<f64> = Vec::new();

        for i in 0..a.rows {
            for j in 0..b.cols {
                let mut sum: f64 = 0.0;
                for k in 0..a.cols {
                    sum += a.get(i, k) * b.get(k, j);
                }
                res.push(sum);
            }
        }

        Matrix::new(a.rows, b.cols, res)
    }

    pub fn add_row(m: &Matrix, s: &Matrix) -> Matrix {
        if s.rows != 1 {
            panic!("Expected a single row matrix");
        }

        if s.cols != m.cols {
            panic!("Matrix columns doesn't match single row matrix columns");
        }

        let mut res: Vec<f64> = Vec::new();

        for i in 0..m.rows {
            for j in 0..m.cols {
                res.push(m.get(i, j) + s.get(0, j));
            }
        }

        Matrix::new(m.rows, m.cols, res)
    }

    pub fn transpose(m: &Matrix) -> Matrix {
        let mut res: Vec<f64> = Vec::new();

        for j in 0..m.cols {
            for i in 0..m.rows {
                res.push(m.get(i, j))
            }
        }

        Matrix::new(m.cols, m.rows, res)
    }
}
