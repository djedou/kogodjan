

/// Saturating Linear
pub fn satlin(input: f64) -> f64 {
    if input < 0.0 {
        0.0
    }
    else if 0.0 <= input && input <= 1.0 {
        input
    }
    else {
        1.0
    }
}