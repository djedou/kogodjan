use crate::maths::{
    types::MatrixD
};

pub fn sgd(lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, param: &MatrixD<f64>, input: Option<&MatrixD<f64>>) -> MatrixD<f64> {
    let mut param_clone = param.clone();
    
    if let Some(inp) = input {
        //let inp_sum = inp;
        let inp_sum_t = inp.transpose();
        let mut grad_input = MatrixD::<f64>::zeros(gradient.nrows(), inp_sum_t.ncols());
        gradient.mul_to(&inp_sum_t, &mut grad_input);
        param_clone.zip_apply(&grad_input, |p, g| p - (lr * (g / (*batch_size as f64))));
        
        param_clone
    } else {
        let grad_sum = gradient.column_sum();
        param_clone.zip_apply(&grad_sum, |p, g| p - (lr * (g / (*batch_size as f64))));
        
        param_clone
    }
}
