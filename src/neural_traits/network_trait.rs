use crate::loss_functions::{LossFunction, GradFunction};
use crate::maths::Matrix;

pub trait NetworkT {

    fn train(&mut self, lr: f64, batch_size: usize, optimizers: (LossFunction, GradFunction), epoch: i32) -> Result<(), String>;
    fn predict(&mut self, input: &Matrix<f64>) -> Result<Matrix<f64>, String>;
}
