use djed_maths::linear_algebra::matrix::Matrix;

/// rectified linear unit
pub fn relu(input: Matrix<f64>) -> Matrix<f64> {
    //1.0 / (1.0 + (-input).exp());
    let mut res = input.clone();
    //res.apply(|a| a.max(0.0));
    res
}

/// derivative of rectified linear unit
pub fn relu_deriv(input: Matrix<f64>) -> Matrix<f64> {
    let mut res = input.clone();
    /*res.apply(|a| {
        if a < 0.0 {
            0.0
        }
        else if a > 0.0 {
            1.0
        }
        else {
            0.0
        }
    });*/

    res
}
