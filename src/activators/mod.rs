mod sigmoid;
mod tanh;

use sigmoid::{sigmoid, sigmoid_derivative};
use tanh::{tanh, tanh_derivative};
use serde::{Deserialize, Serialize};



/// Defines the activation of a layer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Tanh
}


impl Activation {

    pub fn run(&self, z: f64) -> f64 {
        return match self {
            Self::Sigmoid => sigmoid(z),
            Self::Tanh => tanh(z)
        };
        
    }
    
    pub fn derivative(&self, z: f64) -> f64 {
        return match self {
            Self::Sigmoid => sigmoid_derivative(z),
            Self::Tanh => tanh_derivative(z),
        };
    }
}