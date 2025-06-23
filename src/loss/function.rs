use crate::matrix::Matrix;

pub struct LossFn;

impl LossFn {
    pub fn from_onehot(input: &Matrix) -> Vec<usize> {
        let mut res: Vec<usize> = Vec::new();
        for i in 0..input.rows {
            for j in 0..input.cols {
                if input.get(i, j) == 1.0 {
                    res.push(j);
                }
            }
        }
        res
    }

    pub fn calculate(input: &Matrix, target: &Vec<usize>) -> f64 {
        let epsilon: f64 = 1e-15;
        let mut losses: Vec<f64> = Vec::new();

        for i in 0..input.rows {
            let j: usize = target[i];
            let pred = input.get(i, j).clamp(epsilon, 1.0 - epsilon);
            let loss: f64 = pred.ln();
            losses.push(-loss);
        }

        let sum: f64 = losses.iter().sum();
        sum / losses.len() as f64
    }

    pub fn accuracy(input: &Matrix, target: &Vec<usize>) -> f64 {
        let mut res: f64 = 0.0;

        for i in 0..input.rows {
            let mut k: usize = 0;
            let mut m: f64 = input.get(i, k);
            for j in 1..input.cols {
                let n: f64 = input.get(i, j);
                if n > m {
                    m = n;
                    k = j;
                }
            }
            if target[i] == k {
                res += 1.0;
            }
        }

        res / target.len() as f64
    }
}
