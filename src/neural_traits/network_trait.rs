use crate::loss_functions::{LossFunction, GradFunction};
use crate::optimizers::Optimizer;
use djed_maths::linear_algebra::matrix::Matrix;

pub trait NetworkT {

    fn train(&mut self, lr: f64, batch_size: usize, optimizers: (LossFunction, GradFunction, Optimizer), epoch: i32) -> Result<(), String>;
    fn predict(&mut self, input: &Matrix<f64>) -> Result<Matrix<f64>, String>;
}
