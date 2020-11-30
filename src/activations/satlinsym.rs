

/// Symmetric Saturating Linear
pub fn satlinsym(input: f64) -> f64 {
    if input < -1.0 {
        -1.0
    }
    else if -1.0 <= input && input <= 1.0 {
        input
    }
    else {
        1.0
    }
}