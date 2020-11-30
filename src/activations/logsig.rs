
/// Log-Sigmoid
pub fn logsig(input: f64) -> f64 {
    1.0 / (1.0 + (-input).exp())
}