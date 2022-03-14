mod mse;
mod binary_cross_entropy;

use serde::{Deserialize, Serialize};
use ndarray::{Array2};
use mse::{mse_loss, mse_derivative};
use binary_cross_entropy::{binary_cross_entropy_loss, binary_cross_entropy_derivative};



#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Loss {
    Mse,
    BinaryCrossEntropy
}

impl Loss {
    pub fn run(&self, output: &[Array2<f64>], target: &[Array2<f64>]) -> Vec<f64> {
        return match self {
            Self::Mse => mse_loss(&output, &target),
            BinaryCrossEntropy => binary_cross_entropy_loss(&output, &target)
        };
        
    }
    
    pub fn derivative(&self, outputs: Vec<Array2<f64>>, targets: &[Array2<f64>]) -> Vec<Array2<f64>> {
        return match self {
            Self::Mse => mse_derivative(&outputs, &targets),
            BinaryCrossEntropy => binary_cross_entropy_derivative(&outputs, &targets)
        };
    }
}