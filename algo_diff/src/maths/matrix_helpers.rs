use crate::maths::Matrix;
use rayon::prelude::*;

// Vec<(f64, f64)> is a Row
pub type Details = Vec<Vec<((usize, usize), (f64, f64))>>;

fn matrix_details(lhs: &Matrix, rhs: &Matrix) -> Details {
    let nrows = lhs.nrows();
    let ncols = lhs.ncols();

    (0..nrows)
        .into_par_iter()
        .map(|r| {
            let rows: Vec<((usize, usize), (f64, f64))> = (0..ncols)
            .into_par_iter()
            .map(|c| ((r, c), (lhs[(r, c)], rhs[(r, c)])))
            .collect();
            rows
        })
        .collect()
}


pub fn get_div_deriv(lhs: &Matrix, rhs: &Matrix) -> [Matrix; 2] {

    let details = matrix_details(&lhs, &rhs);
    
    let mut lhs_matrix = Matrix::zeros((lhs.nrows(), lhs.ncols()));
    div_deriv(&mut lhs_matrix, &details, true);
    let mut rhs_matrix = Matrix::zeros((rhs.nrows(), rhs.ncols()));
    div_deriv(&mut rhs_matrix, &details, false);

    [lhs_matrix, rhs_matrix]
}

pub fn div_deriv(deriv: &mut Matrix, inputs: &Details, is_lhs: bool) {
    inputs
        .into_iter()
        .for_each(|r| {
            r.into_iter()
                .for_each(|d| {
                    deriv[(d.0.0, d.0.1)] = {
                        let res: f64 = if is_lhs {if d.1.1 == 0. {0.} else {1. / d.1.1}} else {if d.1.1 == 0. {-0.} else { -d.1.0 / (d.1.1).powi(2)}};
                        if res.is_nan() {0.} else {res}
                    }
                });
        });
}


#[cfg(test)]
mod maths_helpers_test {
    use crate::maths::Matrix;
    use super::*;


    #[test]
    fn add() {
        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();

        println!("lhs: {:?}", lhs);
        println!("rhs: {:?}", rhs);
        let details = matrix_details(&lhs, &rhs);
        println!("details: {:?}", details);
    }

    #[test]
    fn matrix_dot() {
        let _lhs = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        let _rhs = Matrix::from_shape_vec((2,4), vec![1., 2., 1., 1., 1., 2., 0., 1.]).unwrap();
    }

}