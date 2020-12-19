use crate::maths::types::MatrixD;

/// Log-Sigmoid
pub fn logsig(input: MatrixD<f32>) -> MatrixD<f32> {
    //1.0 / (1.0 + (-input).exp());
    let mut res = input.clone();
    res.apply(|a| 1.0 / (1.0 + (-a).exp()));
    res
}


/// derivative of Log-Sigmoid
pub fn logsig_deriv(input: MatrixD<f32>) -> MatrixD<f32> {
    //1.0 / (1.0 + (-input).exp());
    let mut res = input.clone();
    res.apply(|a| {
        let b = 1.0 / (1.0 + (-a).exp());
        (1.0 - b) * b
    });
    res
}