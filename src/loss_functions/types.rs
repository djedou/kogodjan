use crate::maths::types::MatrixD;

pub type LossFunction = fn(output: &MatrixD<f32>, target: &MatrixD<f32>) -> MatrixD<f32>; 
pub type GradFunction = fn(errors: MatrixD<f32>) -> MatrixD<f32>; 