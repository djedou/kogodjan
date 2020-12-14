use crate::maths::types::MatrixD;

/// Log-Sigmoid
pub fn logsig(input: MatrixD<f64>) -> MatrixD<f64> {
    //1.0 / (1.0 + (-input).exp());
    let mut res = input.clone();
    res.apply(|a| 1.0 / (1.0 + (-a).exp()));
    res
}