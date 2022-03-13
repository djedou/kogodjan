use ndarray::{Array2};


pub fn mse_loss(output: &Array2<f64>, target: &Array2<f64>) -> Vec<f64> {
    let mut res = Vec::new();

    let loss = |out: &[f64], targ: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..targ.len() {
            sum += (targ[i] - out[i]).exp2();
        }
        sum / targ.len() as f64
    };
    
    for i in 0..target.ncols() {
        res.push(loss(&output.column(i).to_vec(), &target.column(i).to_vec()));
    }

    res
}

pub fn binary_cross_entropy_loss(output: &Array2<f64>, target: &Array2<f64>) -> Vec<f64> {
    let mut res = Vec::new();

    let loss = |out: &[f64], targ: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..targ.len() {
            sum += (targ[i] * out[i].ln()) + ((1.0 - targ[i]) * (1.0 - out[i]).ln());
        }
        (-1.0 * sum) / targ.len() as f64
    };
    
    for i in 0..target.ncols() {
        res.push(loss(&output.column(i).to_vec(), &target.column(i).to_vec()));
    }

    res
}

pub fn mse_derivative(output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
    let res = target - output;
    let menos_two = Array2::from_shape_fn((target.nrows(), target.ncols()), |_| 2.0);
    
    menos_two * res
}

pub fn binary_cross_entropy_derivative(output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
    let res = target - output;
    let menos_two = Array2::from_shape_fn((target.nrows(), target.ncols()), |_| 2.0);
    
    menos_two * res
}