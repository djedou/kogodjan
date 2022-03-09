//use crate::loss_functions::{LossFunction, GradFunction};
use crate::maths::Matrix;

pub trait NetworkT {

    fn train(&mut self, lr: f64, batch_size: usize, epoch: i32);
    fn predict(&mut self, input: &[f64]) -> Matrix<f64>;
}
