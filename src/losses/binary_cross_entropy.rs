use ndarray::{Array2, Array};


pub fn binary_cross_entropy_loss(output: &[Array2<f64>], target: &[Array2<f64>]) -> Vec<f64> {
    let mut res = Vec::new();

    let loss = |out: &[f64], targ: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..targ.len() {
            sum += (targ[i] * out[i].ln()) + ((1.0 - targ[i]) * (1.0 - out[i]).ln());
        }
        (-1.0 * sum) / targ.len() as f64
    };
    
    for i in 0..target.len() {
        res.push(loss(&output[i].column(0).to_vec(), &target[i].column(0).to_vec()));
    }

    res
}

pub fn binary_cross_entropy_derivative(outputs: &[Array2<f64>], targets: &[Array2<f64>]) -> Vec<Array2<f64>> {
    let cal = |o: f64, t: f64| -> f64 {
        (-1.0 * (t/o)) + ((1.0 - t)/(1.0 - o))
    };

    let deriv = |out: &[f64], tar: &[f64]| -> Vec<f64> {
        out.iter().zip(tar.iter()).map(|(o,t)| -> f64 {cal(*o,*t)}).collect()
    };
    
    let mut deriv_arr: Vec<Array2<f64>> = vec![];

    for i in 0..outputs.len() {
        let col = deriv(&outputs[i].column(0).to_vec(), &targets[i].column(0).to_vec());
        deriv_arr.push(Array::from_vec(col.clone()).into_shape((col.len(), 1)).unwrap());     
    }
    
    deriv_arr
}