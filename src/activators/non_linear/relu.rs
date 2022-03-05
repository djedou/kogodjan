use crate::maths::Matrix;

/// rectified linear unit
pub fn relu(input: Matrix<f64>) -> Matrix<f64> {
    let res = input.clone();
    res
}

/// derivative of rectified linear unit
pub fn relu_deriv(input: Matrix<f64>) -> Matrix<f64> {
    let res = input.clone();

    res
}
