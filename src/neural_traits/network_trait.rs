use crate::loss_functions::{LossFunction, GradFunction};
use crate::optimizers::Optimizer;
use crate::maths::types::MatrixD;

pub trait NetworkT {

    fn train(&mut self, lr: f32, batch_size: Option<usize>, optimizers: (LossFunction, GradFunction, Optimizer), epoch: i32);
    fn predict(&mut self, input: &MatrixD<f32>) -> MatrixD<f32>;
}