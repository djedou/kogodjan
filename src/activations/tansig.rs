
/// Hyperbolic Tangent Sigmoid
pub fn tansig(input: f64) -> f64 {
    (input.exp() - (-input).exp()) / (input.exp() + (-input).exp())
}