use crate::maths::Matrix;

pub type LossFunction = fn(output: &Matrix<f64>, target: &Matrix<f64>) -> f64; 
pub type GradFunction = fn(output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64>; 
