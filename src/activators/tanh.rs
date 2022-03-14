

pub fn tanh(x: f64) -> f64{
    x.tanh()
}

pub fn tanh_derivative(z: f64) -> f64 {
    1.0 - z.exp2()
}