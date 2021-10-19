use djed_maths::linear_algebra::matrix::Matrix;

/// Hyperbolic tangent Sigmoid `tansig` or `tanh`
pub fn tansig(input: Matrix<f64>) -> Matrix<f64> {
    //1.0 / (1.0 + (-input).exp());
    let mut res = input.clone();
    //res.apply(|a| (a.exp() - (-a).exp()) / (a.exp() + (-a).exp()));
    res
}


/// derivative of Hyperbolic tangent Sigmoid
pub fn tansig_deriv(input: Matrix<f64>) -> Matrix<f64> {
    //1.0 / (1.0 + (-input).exp());
    let mut res = input.clone();
    /*res.apply(|a| {
        let b = (a.exp() - (-a).exp()) / (a.exp() + (-a).exp());
        1.0 - b.exp2()
    });*/
    res
}
