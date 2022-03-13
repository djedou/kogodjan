use ndarray::{Array2, Array};


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

pub fn binary_cross_entropy_derivative(output: &Array2<f64>, target: &Array2<f64>) -> Array2<f64> {
    let cal = |o: f64, t: f64| -> f64 {
        (-1.0 * (t/o)) + ((1.0 - t)/(1.0 - o))
    };

    let deriv = |out: &[f64], tar: &[f64]| -> Vec<f64> {
        out.iter().zip(tar.iter()).map(|(o,t)| -> f64 {cal(*o,*t)}).collect()
    };
    
    let mut deriv_arr = Array2::<f64>::zeros((output.nrows(), 0));

    for i in 0..output.ncols() {
        let col = deriv(&output.column(i).to_vec(), &target.column(i).to_vec());
        deriv_arr.push_column(Array::from_vec(col).view()).unwrap();     
    }
    
    deriv_arr
}