use crate::maths::Matrix;
//use std::sync::{Arc, Mutex};

/// Squared loss
pub fn squared_loss(_output: &Matrix<f64>, _target: &Matrix<f64>) -> Result<f64, String> {
    
    Ok(0.0)
}

/// Squared loss gradient
pub fn squared_loss_gradient(output: &Matrix<f64>, _target: &Matrix<f64>) -> Matrix<f64> {
    
    output.clone()
}
