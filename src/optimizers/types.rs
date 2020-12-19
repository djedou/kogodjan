use crate::maths::types::{MatrixD};

pub type Optimizer = fn(lr: &f32, batch_size: &usize, gradient: &MatrixD<f32>, param: &MatrixD<f32>, input: Option<&MatrixD<f32>>) -> MatrixD<f32>;
