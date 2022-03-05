use crate::maths::Matrix;

pub type Optimizer = fn(lr: f64, gradient: &Matrix<f64>, param: &Matrix<f64>, input: Option<&Matrix<f64>>) -> Matrix<f64>;
