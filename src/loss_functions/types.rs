use djed_maths::linear_algebra::matrix::Matrix;

pub type LossFunction = fn(output: &Matrix<f64>, target: &Matrix<f64>) -> f64; 
pub type GradFunction = fn(output: &mut Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64>; 
