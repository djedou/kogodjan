use super::{FLOAT_SIZE_10000};
/// Hyperbolic Tangent Sigmoid
pub fn tansig(input: f64) -> f64 {
    let res = (input.exp() - (-input).exp()) / (input.exp() + (-input).exp());
    (res * FLOAT_SIZE_10000).trunc() / FLOAT_SIZE_10000
}

pub fn tansig_deriv(input: f64) -> f64 {
    let res = 1.0 - tansig(input).exp2();
    (res * FLOAT_SIZE_10000).trunc() / FLOAT_SIZE_10000
}