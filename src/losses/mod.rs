mod mse;

use serde::{Deserialize, Serialize};
use ndarray::{Array2};
use mse::{mse_loss, mse_derivative};



#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Loss {
    Mse
}

impl Loss {
    pub fn run(&self, output: &Array2<f64>, target: &Array2<f64>) -> Vec<f64> {
        return match self {
            Self::Mse => mse_loss(&output, &target)
        };
        
    }
    
    pub fn derivative(&self, output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
        return match self {
            Self::Mse => mse_derivative(&output, &target)
        };
    }
}