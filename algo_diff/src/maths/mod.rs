mod matrix_helpers;

use ndarray::linalg::{general_mat_mul};
use ndarray::{ArrayBase, Data, DataMut, Ix2};
use rand::{
    distributions::{Distribution, uniform::Uniform}
};



/// Alias for a `f64` `ndarray` matrix.
pub type Matrix = ndarray::Array2<f64>;
pub use ndarray::{Array, Axis};
pub(crate) use matrix_helpers::*;

/// Uses approximate e^x
#[inline(always)]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

#[inline(always)]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

#[inline(always)]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

#[inline(always)]
pub fn sigmoid(x: f64) -> f64 {
    let critical_value = 10.0;

    if x > critical_value {
        1.0
    } else if x < -critical_value {
        0.0
    } else {
        1.0 / (1.0 + exp(-x))
    }
}

#[inline(always)]
pub fn pow2(x: f64) -> f64 {
    x.powi(2)
}

pub fn softmax_exp_sum(xs: &[f64], max: f64) -> f64 {
    let mut xs = xs;
    let mut s = 0.;

    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);

    while xs.len() >= 8 {
        p0 += exp(xs[0] - max);
        p1 += exp(xs[1] - max);
        p2 += exp(xs[2] - max);
        p3 += exp(xs[3] - max);
        p4 += exp(xs[4] - max);
        p5 += exp(xs[5] - max);
        p6 += exp(xs[6] - max);
        p7 += exp(xs[7] - max);

        xs = &xs[8..];
    }
    s += p0 + p4;
    s += p1 + p5;
    s += p2 + p6;
    s += p3 + p7;

    for i in 0..xs.len() {
        s += exp(xs[i] - max)
    }

    s
}

// matrix multiply
pub fn mat_mul<S1, S2, S3>(
    alpha: f64,
    lhs: &ArrayBase<S1, Ix2>,
    rhs: &ArrayBase<S2, Ix2>,
    beta: f64,
    out: &mut ArrayBase<S3, Ix2>,
) where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    S3: DataMut<Elem = f64>,
{
    general_mat_mul(alpha, lhs, rhs, beta, out);
}

/// SIMD-enabled vector-vector dot product.
pub fn simd_dot(xs: &[f64], ys: &[f64]) -> f64 {
    let len = std::cmp::min(xs.len(), ys.len());
    let mut xs = &xs[..len];
    let mut ys = &ys[..len];

    let mut s = 0.;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);

    while xs.len() >= 8 {
        p0 += xs[0] * ys[0];
        p1 += xs[1] * ys[1];
        p2 += xs[2] * ys[2];
        p3 += xs[3] * ys[3];
        p4 += xs[4] * ys[4];
        p5 += xs[5] * ys[5];
        p6 += xs[6] * ys[6];
        p7 += xs[7] * ys[7];

        xs = &xs[8..];
        ys = &ys[8..];
    }
    s += p0 + p4;
    s += p1 + p5;
    s += p2 + p6;
    s += p3 + p7;

    for i in 0..xs.len() {
        s += xs[i] * ys[i];
    }

    s
}

pub fn simd_sum(xs: &[f64]) -> f64 {
    let mut xs = xs;

    let mut s = 0.;
    let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
        (0., 0., 0., 0., 0., 0., 0., 0.);

    while xs.len() >= 8 {
        p0 += xs[0];
        p1 += xs[1];
        p2 += xs[2];
        p3 += xs[3];
        p4 += xs[4];
        p5 += xs[5];
        p6 += xs[6];
        p7 += xs[7];

        xs = &xs[8..];
    }

    s += p0 + p4;
    s += p1 + p5;
    s += p2 + p6;
    s += p3 + p7;

    for i in 0..xs.len() {
        s += xs[i];
    }

    s
}

pub fn simd_scaled_assign(xs: &mut [f64], ys: &[f64], alpha: f64) {
    for (x, y) in xs.iter_mut().zip(ys.iter()) {
        *x = y * alpha;
    }
}

pub fn simd_scaled_add(xs: &mut [f64], ys: &[f64], alpha: f64) {
    for (x, y) in xs.iter_mut().zip(ys.iter()) {
        *x += y * alpha;
    }
}

/*
/// Return a Xavier-normal initialised random array.
pub fn xavier_normal(rows: usize, cols: usize) -> Matrix {
    let normal = Normal::new(0.0, 1.0 / (rows as f64).sqrt());
    Matrix::zeros((rows, cols)).map(|_| normal.sample(&mut rand::thread_rng()) as f64)
}
*/
/// Return a random matrix with values drawn uniformly from `(min, max)`.
pub fn uniform<R: rand::Rng>(rows: usize, cols: usize, min: f64, max: f64, rng: &mut R) -> Matrix {
    let dist = Uniform::new(min, max);
    Matrix::zeros((rows, cols)).map(|_| dist.sample(rng) as f64)
}


/*
fn row_wise_stack(dest: &mut Matrix, lhs: &Matrix, rhs: &Matrix) {
    for (mut dest_row, source_row) in dest
        .rows_mut()
        .into_iter()
        .zip(lhs.rows().into_iter().chain(rhs.rows()))
    {
        slice_assign(
            dest_row.as_slice_mut().unwrap(),
            source_row.as_slice().unwrap(),
        );
    }
}

fn column_wise_stack(dest: &mut Matrix, lhs: &Matrix, rhs: &Matrix) {
    for (mut dest_row, lhs_row, rhs_row) in izip!(
        dest.rows_mut().into_iter(),
        lhs.rows().into_iter(),
        rhs.rows().into_iter()
    ) {
        let dest_row = dest_row.as_slice_mut().unwrap();
        let lhs_row = lhs_row.as_slice().unwrap();
        let rhs_row = rhs_row.as_slice().unwrap();

        let (left, right) = dest_row.split_at_mut(lhs_row.len());
        slice_assign(left, lhs_row);
        slice_assign(right, rhs_row);
    }
}


fn column_wise_stack_gradient(gradient: &Matrix, lhs: &mut Matrix, rhs: &mut Matrix, op: &BackwardAction) {
    for (grad_row, mut lhs_row, mut rhs_row) in izip!(
        gradient.rows().into_iter(),
        lhs.rows_mut().into_iter(),
        rhs.rows_mut().into_iter()
    ) {
        let grad_row = grad_row.fast_slice();
        let lhs_row = lhs_row.fast_slice_mut();
        let rhs_row = rhs_row.fast_slice_mut();

        let (left, right) = grad_row.split_at(lhs_row.len());

        match op {
            &BackwardAction::Increment => {
                for (x, y) in lhs_row.iter_mut().zip(left.iter()) {
                    *x += y;
                }
                for (x, y) in rhs_row.iter_mut().zip(right.iter()) {
                    *x += y;
                }
            }
            &BackwardAction::Set => {
                lhs_row.copy_from_slice(left);
                rhs_row.copy_from_slice(right);
            }
        }
    }
}

fn row_wise_stack_gradient(gradient: &Matrix, lhs: &mut Matrix, rhs: &mut Matrix, op: &BackwardAction) {
    for (grad_row, mut dest_row) in gradient
        .rows()
        .into_iter()
        .zip(lhs.rows_mut().into_iter().chain(rhs.rows_mut()))
    {
        let grad_row = grad_row.as_slice().unwrap();
        let dest_row = dest_row.as_slice_mut().unwrap();

        match op {
            &BackwardAction::Increment => for (x, y) in dest_row.iter_mut().zip(grad_row.iter()) {
                *x += y;
            },
            &BackwardAction::Set => for (x, &y) in dest_row.iter_mut().zip(grad_row.iter()) {
                *x = y;
            },
        }
    }
}
*/

/*
#[cfg(test)]
mod tests {

    use std;

    use super::*;

    use rand;
    use rand::Rng;

    use nn;

    fn random_matrix(rows: usize, cols: usize) -> Arr {
        nn::xavier_normal(rows, cols)
    }

    fn array_scaled_assign(xs: &mut Arr, ys: &Arr, alpha: f64) {
        for (x, y) in xs.iter_mut().zip(ys.iter()) {
            *x = y * alpha;
        }
    }

    fn scaled_assign(xs: &mut Arr, ys: &Arr, alpha: f64) {
        // assert_eq!(xs.shape(), ys.shape(), "Operands do not have the same shape.");

        let xs = xs.as_slice_mut().expect("Unable to convert LHS to slice.");
        let ys = ys.as_slice().expect("Unable to convert RHS to slice.");

        simd_scaled_assign(xs, ys, alpha);
    }

    fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
        lhs.iter().zip(rhs.iter()).map(|(x, y)| x * y).sum()
    }

    fn unrolled_dot(xs: &[f64], ys: &[f64]) -> f64 {
        let len = std::cmp::min(xs.len(), ys.len());
        let mut xs = &xs[..len];
        let mut ys = &ys[..len];

        let mut s = 0.;
        let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) =
            (0., 0., 0., 0., 0., 0., 0., 0.);

        while xs.len() >= 8 {
            p0 += xs[0] * ys[0];
            p1 += xs[1] * ys[1];
            p2 += xs[2] * ys[2];
            p3 += xs[3] * ys[3];
            p4 += xs[4] * ys[4];
            p5 += xs[5] * ys[5];
            p6 += xs[6] * ys[6];
            p7 += xs[7] * ys[7];

            xs = &xs[8..];
            ys = &ys[8..];
        }
        s += p0 + p4;
        s += p1 + p5;
        s += p2 + p6;
        s += p3 + p7;

        for i in 0..xs.len() {
            s += xs[i] * ys[i];
        }

        s
    }

    #[test]
    fn test_fastexp() {
        let values: Vec<f64> = vec![-0.5, -0.1, 0.0, 0.1, 0.5];
        for &x in &values {
            println!("Input: {}, stdlib: {}, fast: {}", x, x.exp(), fastexp(x));
        }
    }

    #[test]
    fn test_fastlog() {
        let values: Vec<f64> = vec![0.1, 0.5, 1.0, 5.0, 10.0];
        for &x in &values {
            println!("Input: {}, stdlib: {}, fast: {}", x, x.ln(), fastlog(x));
        }
    }

    #[test]
    fn test_tanh() {
        let values: Vec<f64> = vec![-0.5, -0.1, 0.0, 0.1, 0.5];
        for &x in &values {
            println!(
                "Input: {}, stdlib: {}, fast: {}",
                x,
                x.tanh(),
                tanhf_fast(x)
            );
        }
    }

    #[test]
    fn test_dot() {
        for len in 0..32 {
            let xs = (0..len)
                .map(|_| rand::thread_rng().gen())
                .collect::<Vec<f64>>();
            let ys = (0..len)
                .map(|_| rand::thread_rng().gen())
                .collect::<Vec<f64>>();

            let _dot = dot(&xs[..], &ys[..]);
            let _unrolled_dot = unrolled_dot(&xs[..], &ys[..]);
            let _simd_dot = simd_dot(&xs[..], &ys[..]);

            let epsilon = 1e-5;

            assert!((_dot - _unrolled_dot).abs() < epsilon);
            assert!((_dot - _simd_dot).abs() < epsilon, "{} {}", _dot, _simd_dot);
        }
    }

    #[test]
    fn test_scaled_assign() {
        for len in 0..32 {
            let mut xs_1 = random_matrix(len, 1);
            let mut xs_2 = xs_1.clone();
            let ys = random_matrix(len, 1);

            let alpha = 3.5;

            array_scaled_assign(&mut xs_1, &ys, alpha);
            scaled_assign(&mut xs_2, &ys, alpha);

            assert_eq!(xs_1, xs_2);
        }
    }

    #[allow(dead_code)]
    fn assert_close(x: &Arr, y: &Arr, tol: f64) {
        assert!(
            x.all_close(y, tol),
            "{:#?} not within {} of {:#?}",
            x,
            tol,
            y
        );
    }

    #[test]
    fn test_dot_node_specializations_mm() {
        let x = random_matrix(64, 64);
        let y = random_matrix(64, 64);

        let mut result = random_matrix(64, 64);
        let mut expected = random_matrix(64, 64);

        mat_mul(1.0, &x, &y, 0.0, &mut result);
        general_mat_mul(1.0, &x, &y, 0.0, &mut expected);

        assert_close(&result, &expected, 0.001);
    }

    #[test]
    fn test_dot_node_specializations_mv() {
        let x = random_matrix(64, 64);
        let y = random_matrix(64, 1);

        let mut result = random_matrix(64, 1);
        let mut expected = random_matrix(64, 1);

        mat_mul(1.0, &x, &y, 0.0, &mut result);
        general_mat_mul(1.0, &x, &y, 0.0, &mut expected);

        assert_close(&result, &expected, 0.001);
    }

    #[test]
    fn test_dot_node_specializations_vm() {
        let x = random_matrix(1, 64);
        let y = random_matrix(64, 64);

        let mut result = random_matrix(1, 64);
        let mut expected = random_matrix(1, 64);

        mat_mul(1.0, &x, &y, 0.0, &mut result);
        general_mat_mul(1.0, &x, &y, 0.0, &mut expected);

        assert_close(&result, &expected, 0.001);
    }
}
*/