use djed_maths::linear_algebra::{
    matrix::Matrix,
    //vector::ZipCallBack
};
use std::sync::{Arc, Mutex};

/// Squared loss
pub fn squared_loss(output: &Matrix<f64>, target: &Matrix<f64>, bat_size: usize) -> Result<f64, String> {
    let callback = |t,o| (t as f64) - (o as f64);
    let cost = target.zip_apply(&output, callback)?;
    let mut lost = 0.0;
    for i in 0..cost.ncols() {
        let norm = cost.get_col(i).norm().exp2();
        lost = lost + (0.5 * norm);
    }

    let res = lost / (bat_size as f64);
    Ok(res)
}

/// Squared loss gradient
pub fn squared_loss_gradient(output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> {
    // o - t
    let callback = |o,t| (o as f64) - (t as f64);
    let cost = output.zip_apply(&target, callback).unwrap();
    
    cost
}
