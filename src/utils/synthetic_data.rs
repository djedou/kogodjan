use rand::{thread_rng, Rng};
use crate::maths::types::MatrixD;
use nalgebra::{DVector};


pub fn synthetic_data(w: &MatrixD<f32>, b: f32, num_examples: usize) -> (MatrixD<f32>, MatrixD<f32>) {
    // Generate y = wx + b + noise
    let mut rng = thread_rng(); 
    
    let x = MatrixD::<f32>::from_fn(w.len(), num_examples, |_a, _b| {
            
        let value = rng.gen::<f32>(); // generate float between 0.0 and 1.0
        value
    });
    let (wr, _wc) = w.shape();
    let (_xr, xc) = x.shape();

    let mut dest = MatrixD::<f32>::zeros(wr, xc);

    w.mul_to(&x, &mut dest);
    let y1 = dest.add_scalar(b);
    let (y1r, y1c) = y1.shape();

    let y2 = MatrixD::<f32>::from_fn(y1r, y1c, |_a, _b| {
            
        rng.gen::<f32>() // generate float between 0.0 and 1.0
    });

    let mut y = MatrixD::<f32>::zeros(y1r, y1c);
    y1.add_to(&y2, &mut y);

    (x,y)

}

pub fn synthetic_data_mat(w: &MatrixD<f32>, bias: MatrixD<f32>, num_examples: usize) -> (MatrixD<f32>, MatrixD<f32>) {
    // Generate y = wx + b + noise
    let mut rng = thread_rng(); 

    let x = MatrixD::<f32>::from_fn(w.ncols(), num_examples, |_a, _b| {
            
        let value = rng.gen::<f32>(); // generate float between 0.0 and 1.0
        value
    });

    let (wr, _wc) = w.shape();
    let (_xr, xc) = x.shape();

    // calculate net_inputs
    let mut dest = MatrixD::<f32>::zeros(wr, xc);

    w.mul_to(&x, &mut dest);
    let mut y1 = MatrixD::zeros(wr, xc);

    (0..xc).into_iter().for_each(|m| {
        let mut col_value = DVector::<f32>::from_element(wr, 0.0);
        dest.column(m).add_to(&bias, &mut col_value);
        y1.set_column(m, &col_value);
        
    });

    // calculate noises
    let (y1r, y1c) = y1.shape();

    let y2 = MatrixD::<f32>::from_fn(y1r, y1c, |_a, _b| {
            
        rng.gen::<f32>() // generate float between 0.0 and 1.0
    });

    let mut y = MatrixD::<f32>::zeros(y1r, y1c);
    y1.add_to(&y2, &mut y);

    (x,y)

}