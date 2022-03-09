use serde::{Deserialize, Serialize};


/// Defines the activation of a layer.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Activation {
    /// Sigmoid activation functions.
    ///
    /// $ A(z)=\frac{1}{1+e^{-z}} $
    Sigmoid,
    /// Tanh activation functions.
    ///
    /// $ A(z)=\frac{2}{1+e^{-2z}}-1 $
    Tanh,
}

impl Activation {
    /// Computes activations given inputs (A(z)).
    pub fn run(&self, z: f64) -> f64 {
        return match self {
            Self::Sigmoid => self.sigmoid(z),
            Self::Tanh => self.tanh(z)
        };
        
    }
    
    fn sigmoid(&self, z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    fn tanh(&self, x: f64) -> f64{
        x.tanh()
    }

    /// Derivative w.r.t. layer input (∂a/∂z).
    pub fn derivative(&self, z: f64) -> f64 {
        // What should we name the derivative functions?
        return match self {
            Self::Sigmoid => self.sigmoid_derivative(z),
            Self::Tanh => self.tanh_derivative(z),
        };
    }
    
    // Derivative of sigmoid
    // s' = s(1-s)
    fn sigmoid_derivative(&self, z: f64) -> f64 {
        let s = self.sigmoid(z);
        return s * (1.0 - s);
    }

    // Derivative of tanh
    // t' = 1-t^2
    fn tanh_derivative(&self, z: f64) -> f64 {
        1.0 - z.exp2()
    }
}