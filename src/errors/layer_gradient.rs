use crate::activations::{FLOAT_SIZE_10000};

pub fn others_layers_gradient(deri: f64, error: f64) -> f64 {
    (deri * error * FLOAT_SIZE_10000).trunc() / FLOAT_SIZE_10000
}

pub fn last_layer_gradient(deri: f64, error: f64) -> f64 {
    ( -2.0 * deri * error * FLOAT_SIZE_10000).trunc() / FLOAT_SIZE_10000
}

pub type Gradient = fn(deri: f64, error: f64) -> f64;