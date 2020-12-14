use crate::maths::types::MatrixD;

pub type Activator = fn(MatrixD<f64>) -> MatrixD<f64>;
pub type ActivatorDeriv = fn(MatrixD<f64>) -> MatrixD<f64>;