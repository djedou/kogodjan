use crate::maths::{
    types::MatrixD,
    constants::{FLOAT_SIZE_10000, FLOAT_SIZE_100}
};

pub fn sgd(lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, param: &MatrixD<f64>) -> MatrixD<f64> {
    let sum = (gradient.sum() * FLOAT_SIZE_100).trunc() / FLOAT_SIZE_10000;
    let grad = - (lr * sum) / (*batch_size as f64);
    param.add_scalar(grad)
}