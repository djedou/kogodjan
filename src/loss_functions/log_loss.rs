use crate::maths::Matrix;
use ndarray::{Array2, Zip};

/// log loss
pub fn log_loss(output: &Matrix<f64>, target: &Matrix<f64>) -> f64 {

    let mut res: Array2<f64> = Array2::zeros(output.get_data().dim());

    let rows = res.nrows();
    let cols = res.ncols();

    Zip::from(&mut res)
        .and(&output.get_data())
        .and(&target.get_data())
        .par_for_each(|r, &o, &t| {
            *r = -(t * o.ln() + ((1. - t) * ((1. - o) as f64).ln()));
        });

    res.sum() / (rows * cols) as f64
}

/// Squared loss gradient
pub fn log_loss_gradient(output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
    // o - t
    let mut res: Array2<f64> = Array2::zeros(output.get_data().dim());

    Zip::from(&mut res)
        .and(&output.get_data())
        .and(&target.get_data())
        .par_for_each(|r, &o, &t| {
            *r = o - t;
        });

    Matrix::new_from_array2(&res)

}