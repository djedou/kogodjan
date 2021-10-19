//use crate::maths::types::MatrixD;
//use nalgebra::{DVector};
use djed_maths::linear_algebra::matrix::Matrix;
use std::sync::{Arc, Mutex};
//use crate::utils::round;

/*
pub fn synthetic_data(w: &Matrix<f64>, b: f64, num_examples: usize) -> (Matrix<f64>, Matrix<f64>) {
    // Generate y = wx + b + noise
    let mut rng = thread_rng(); 
    
    let x = MatrixD::<f64>::from_fn(w.len(), num_examples, |_a, _b| {
            
        let value = rng.gen::<f64>(); // generate float between 0.0 and 1.0
        value
    });
    let (wr, _wc) = w.shape();
    let (_xr, xc) = x.shape();

    let mut dest = MatrixD::<f64>::zeros(wr, xc);

    w.mul_to(&x, &mut dest);
    let y1 = dest.add_scalar(b);
    let (y1r, y1c) = y1.shape();

    let y2 = MatrixD::<f64>::from_fn(y1r, y1c, |_a, _b| {
            
        rng.gen::<f64>() // generate float between 0.0 and 1.0
    });

    let mut y = MatrixD::<f64>::zeros(y1r, y1c);
    y1.add_to(&y2, &mut y);

    (x,y)

}
*/
pub fn synthetic_data_mat(w: &Matrix<f64>, bias: Matrix<f64>, num_examples: usize) -> (Matrix<f64>, Matrix<f64>) {
    // Generate y = wx + b + noise
    let callback = |_x| {
        let value: f64 = random!(); // generate float between 0.0 and 1.0
        value
    };

    let x = Matrix::new_from_fn(num_examples, w.nrows(), callback);
    //let w = w.transpose();
    let mut wx = x.dot_product(&w).unwrap();

    // prepared biase
    let mut new_bias = bias.clone();
    for _i in (0..wx.nrows()-1).into_iter() {
        new_bias.add_row(&bias.into_vector().get_data()).unwrap();
    }
    // wx + b
    let mut wx_b = wx.add_matrix(&new_bias).unwrap();
    
    // get noise
    let noise_callback = |_x| {
        let value: f64 = random!(); // generate float between 0.0 and 1.0
        value
    };
    let noise = Matrix::new_from_fn(wx_b.nrows(), wx_b.ncols(), noise_callback);

    // wx + b + noise
    let y = wx_b.add_matrix(&noise).unwrap();

    (x,y)
}
