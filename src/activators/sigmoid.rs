


pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}


pub fn sigmoid_derivative(z: f64) -> f64 {
    let s = sigmoid(z);
    return s * (1.0 - s);
}
