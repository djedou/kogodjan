use djed_maths::linear_algebra::{
    matrix::Matrix
};

use crate::loss_functions::GradFunction;

pub fn extern_gradient(loss_grad_f: &GradFunction,  output: &mut Matrix<f64>, target: &Matrix<f64>) -> Result<Matrix<f64>, String> {
    
    let loss_grad = loss_grad_f(output, target);
    Ok(loss_grad)
}