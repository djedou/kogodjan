use djed_maths::linear_algebra::{
    matrix::Matrix,
    //vector::{ZipCallBack}
};
use std::sync::{Arc, Mutex};


pub fn sgd(lr: f64, gradient: &Matrix<f64>, param: &Matrix<f64>, input: Option<&Matrix<f64>>) -> Matrix<f64> {
    
    let callback = |p, g| p - g;
  
    if let Some(inp) = input {
        let batch_m_lr = lr * (-1.0 / (inp.nrows() as f64));

        let grad = inp.transpose().dot_product(&gradient).unwrap();
        grad.mul_by_scalar(batch_m_lr);

        let new_param = param.zip_apply(&grad, callback).unwrap();

        new_param

    } else {

        let mut param_clone = param.clone();
        let batch_m_lr = lr * (-1.0 / (gradient.nrows() as f64));
        
        for batch in 0..gradient.nrows() {
            let grad = gradient.get_row(batch).into_matrix(gradient.ncols());
            param_clone = param_clone.zip_apply(&grad, callback.clone()).unwrap();
        }
        param_clone.mul_by_scalar(batch_m_lr);

        param_clone
    }
    
}
