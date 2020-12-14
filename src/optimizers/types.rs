use crate::maths::types::MatrixD;

pub type Optimizer = fn(lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, param: &MatrixD<f64>) -> MatrixD<f64>;
