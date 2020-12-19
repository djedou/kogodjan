use crate::maths::types::MatrixD;

pub type Activator = fn(MatrixD<f32>) -> MatrixD<f32>;
pub type ActivatorDeriv = fn(MatrixD<f32>) -> MatrixD<f32>;