use crate::maths::types::MatrixD;

/// rectified linear unit
pub fn relu(input: MatrixD<f64>) -> MatrixD<f64> {
    //1.0 / (1.0 + (-input).exp());
    let mut res = input.clone();
    res.apply(|a| a.max(0.0));
    res
}

/// derivative of rectified linear unit
pub fn relu_deriv(input: MatrixD<f64>) -> MatrixD<f64> {
    let mut res = input.clone();
    res.apply(|a| {
        if a < 0.0 {
            0.0
        }
        else if a > 0.0 {
            1.0
        }
        else {
            0.0
        }
    });

    res
}