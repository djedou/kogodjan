use djed_maths::linear_algebra::{
    matrix::Matrix,
    //vector::ZipCallBack
};
use std::sync::{Arc, Mutex};
//use djed_maths::linear_algebra::vector::{CallBack};
/*use crate::{
    utils::round
};*/

/// log loss
pub fn log_loss(output: &Matrix<f64>, target: &Matrix<f64>) -> f64 {
    let size = output.ncols() * output.nrows();
    let callback = |o,t| ((t as f64) * (o as f64).ln()) + ((1.0 - t as f64) * (1.0 - o as f64).ln());
    
    let cost = output.zip_apply(&target, callback).unwrap();
   
    let mut cols: Vec<f64> = Vec::new();
    for i in 0..cost.ncols() {
        let col: f64 = cost.get_col(i).get_data().into_iter().sum();
        cols.push(col);
    }
    let network_cost: f64 = cols.into_iter().sum();
    let network_cost: f64 = network_cost * (- 1.0 / size as f64);

    network_cost
}

/// Squared loss gradient
pub fn log_loss_gradient(output: &mut Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
    // o - t
    //let callback = |o,t| (t / o) - ((1.0 - t) / (1.0 - o));
    //let callback: ZipCallBack<f64,f64> = Arc::new(Mutex::new(|o,t| ((1.0 - t) / (1.0 - o)) - (t / o)));
    //let gradient = output.zip_apply(&target, callback).unwrap();
    let gradient = output.sub_matrix(&target).unwrap();
    gradient
}