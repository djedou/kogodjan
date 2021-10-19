
use djed_maths::linear_algebra::{
    matrix::Matrix,
    //vector::CallBack
};
//use std::sync::{Arc, Mutex};


/// Log-Sigmoid
pub fn logsig(input: Matrix<f64>) -> Matrix<f64> {
    //1.0 / (1.0 + (-input).exp());
    let callback = |v| {
        1.0 / (1.0 + (-(v as f64)).exp())
    };
    
    input.apply(callback)
}


/// derivative of Log-Sigmoid
pub fn logsig_deriv(input: Matrix<f64>) -> Matrix<f64> {
    //a * (1-a)
    let callback = |v| {  
        (1.0 / (1.0 + (-(v as f64)).exp())) * (1.0 - (1.0 / (1.0 + (-(v as f64)).exp())))
    };

    input.apply(callback)
}