use crate::maths::types::MatrixD;

pub type LossFunction = fn(output: &MatrixD<f64>, target: &MatrixD<f64>) -> MatrixD<f64>; 
pub type GradFunction = fn(errors: MatrixD<f64>) -> MatrixD<f64>; 