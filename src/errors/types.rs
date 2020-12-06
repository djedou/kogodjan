use ndarray::ArrayD;

pub type NetworkErrorFunction = fn(target: &ArrayD<f64>, output: ArrayD<f64>) -> (bool, ArrayD<f64>); 