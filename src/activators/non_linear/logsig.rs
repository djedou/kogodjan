
use crate::maths::Matrix;
use ndarray::{Array2, Zip};

/// Log-Sigmoid
pub fn logsig(net_input: Matrix<f64>) -> Matrix<f64> {

    let mut output: Array2<f64> = Array2::zeros(net_input.get_data().dim());
    
    Zip::from(&mut output)
        .and(&net_input.get_data())
        .par_for_each(|o, &a| {
            *o = 1.0 / (1.0 + (-a).exp());
        });
    
    Matrix::new_from_array2(&output)
}


/// derivative of Log-Sigmoid
pub fn logsig_deriv(input: Matrix<f64>) -> Matrix<f64> {
    //a * (1-a)
    let mut output: Array2<f64> = Array2::zeros(input.get_data().dim());
    
    Zip::from(&mut output)
        .and(&input.get_data())
        .par_for_each(|o, &a| {
            *o = (1.0 / (1.0 + (-a).exp())) * (1. - (1.0 / (1.0 + (-a).exp())));
        });
    
    Matrix::new_from_array2(&output)
}