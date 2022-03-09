use serde::{Deserialize, Serialize};
use crate::maths::Matrix;
use ndarray::{Array2};



#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum Loss {
    Squared
}

impl Loss {
    /// Computes activations given inputs (A(z)).
    pub fn run(&self, output: &Matrix<f64>, target: &Matrix<f64>) -> Vec<f64> {
        return match self {
            Self::Squared => self.squared_loss(&output, &target)
        };
        
    }
    
    fn squared_loss(&self, output: &Matrix<f64>, target: &Matrix<f64>) -> Vec<f64> {
        let mut res = Vec::new();

        let loss = |out: &[f64], targ: &[f64]| -> f64 {
            let mut sum = 0.0;
            for i in 0..targ.len() {
                sum += (targ[i] - out[i]).exp2();
            }
            sum
        };
        
        for i in 0..target.get_ncols() {
            res.push(loss(&output.get_data().column(i).to_vec(), &target.get_data().column(i).to_vec()));
        }

        res
    }


    /// Derivative w.r.t. layer input (∂a/∂z).
    pub fn derivative(&self, output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
        return match self {
            Self::Squared => self.squared_loss_derivative(&output, &target)
        };
    }
    
    fn squared_loss_derivative(&self, output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
        let res = target.get_data() - output.get_data();
        let menos_two = Array2::from_shape_fn((target.get_nrows(), target.get_ncols()), |_| -2.0);
        
        Matrix::new_from_array2(&(menos_two * res))
    }
}