use crate::maths::Matrix;


pub type Activator = fn(Matrix<f64>) -> Matrix<f64>;
pub type ActivatorDeriv = fn(Matrix<f64>) -> Matrix<f64>;
