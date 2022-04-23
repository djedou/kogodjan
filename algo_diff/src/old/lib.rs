
/*
pub mod maths;
pub mod node;
pub mod graph;
pub mod optim;

use std::rc::Rc;
use itertools::Itertools;
use node::{ParameterNode, InputNode, Node, AddNode, SubNode, MulNode, DivNode};
use node::{Variable};
use maths::{Matrix};
use std::ops::{Add, Sub, Mul, Div};
use optim::*;


pub(crate) fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x > max {
        max
    } else if x < min {
        min
    } else {
        x
    }
}


fn merge_parameters(
    xs: &[Variable<ParameterNode>],
    ys: &[Variable<ParameterNode>],
) -> Vec<Variable<ParameterNode>> {
    xs.iter()
        .merge_join_by(ys.iter(), |x, y| x.as_ptr().cmp(&y.as_ptr()))
        .map(|either| match either {
            itertools::EitherOrBoth::Left(x) => x,
            itertools::EitherOrBoth::Right(x) => x,
            itertools::EitherOrBoth::Both(x, _) => x,
        })
        .cloned()
        .collect()
}


/// An alias for a node whose concrete type has been erased.
pub type BoxedNode = Rc<dyn Node<Value = Matrix, InputGradient = Matrix>>;


macro_rules! impl_arithmetic_op {
    ($trait:ident, $fn:ident, $node:ident) => {
        impl<LHS, RHS> $trait<Variable<RHS>> for Variable<LHS>
        where
            RHS: Node<Value = Matrix, InputGradient = Matrix>,
            LHS: Node<Value = Matrix, InputGradient = Matrix>,
        {
            type Output = Variable<$node<LHS, RHS>>;
            fn $fn(self, other: Variable<RHS>) -> Self::Output {
                Variable::new(
                    Rc::new($node::new(self.node, other.node)),
                    merge_parameters(&self.parameters, &other.parameters),
                )
            }
        }

        /// The constant will be broadcast to have the same shape
        /// as the LHS.
        impl<LHS> $trait<f64> for Variable<LHS>
        where
            LHS: Node<Value = Matrix, InputGradient = Matrix>,
        {
            type Output = Variable<$node<LHS, InputNode>>;
            fn $fn(self, other: f64) -> Self::Output {
                let constant = InputNode::new(self.value().clone() * 0.0 + other);

                Variable::new(
                    Rc::new($node::new(self.node, constant.node)),
                    merge_parameters(&self.parameters, &constant.parameters),
                )
            }
        }

        /// The constant will be broadcast to have the same shape
        /// as the RHS.
        impl<RHS> $trait<Variable<RHS>> for f64
        where
            RHS: Node<Value = Matrix, InputGradient = Matrix>,
        {
            type Output = Variable<$node<InputNode, RHS>>;
            fn $fn(self, other: Variable<RHS>) -> Self::Output {

                let constant = InputNode::new(other.value().clone() * 0.0 + self);

                Variable::new(
                    Rc::new($node::new(constant.node, other.node)),
                    merge_parameters(&constant.parameters, &other.parameters),
                )
            }
        }
    };
}


impl_arithmetic_op!(Add, add, AddNode);
impl_arithmetic_op!(Sub, sub, SubNode);
impl_arithmetic_op!(Mul, mul, MulNode);
impl_arithmetic_op!(Div, div, DivNode);

*/


/*
/// Assert two arrays are within `tol` of each other.
pub fn assert_close(x: &Matrix, y: &Matrix, tol: f64) {
    assert!(
        x.all_close(y, tol),
        "{:#?} not within {} of {:#?}",
        x,
        tol,
        y
    );
}
*/

/*
#[cfg(test)]
mod tests {

    use ndarray::arr2;

    use optim::{Adagrad, Optimizer, SGD};
    use rand::distributions::{Distribution, Uniform};
    use rand::Rng;
    use rayon::prelude::*;
    use std::sync::Arc;

    use super::optim::Synchronizable;
    use super::*;

    const TOLERANCE: f64 = 0.05;

    fn random_matrix(rows: usize, cols: usize) -> Arr {
        nn::xavier_normal(rows, cols)
    }

    fn random_index(rows: usize) -> usize {
        Uniform::new(0, rows).sample(&mut rand::thread_rng())
    }

    #[test]
    fn test_constant_sub() {
        let mut x = ParameterNode::new(Arr::zeros((10, 10)) + 1.0);
        let mut y = (1.0 - x.clone()) * 2.0;

        assert_eq!(y.value().scalar_sum(), 0.0);
        y.zero_gradient();
        y.forward();
        y.backward(1.0);
        assert_eq!(y.value().scalar_sum(), 0.0);

        let (difference, gradient) = finite_difference(&mut x, &mut y);
        assert_close(&difference, &gradient, TOLERANCE);
    }

    #[test]
    fn parameter_deduplication() {
        let x = ParameterNode::new(random_matrix(1, 1));
        let y = ParameterNode::new(random_matrix(1, 1));

        let z = x + y;
        let z = z.clone() + z.clone();

        assert_eq!(z.parameters().len(), 2);
    }

    #[test]
    fn add_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(1, 1));
        let mut y = ParameterNode::new(random_matrix(1, 1));
        let mut z = x.clone() + y.clone() + x.clone() + x.clone();

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
        let (difference, gradient) = finite_difference(&mut y, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn sub_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(1, 1));
        let mut y = ParameterNode::new(random_matrix(1, 1));
        let z = x.clone() - (y.clone() - x.clone());
        let mut z = z.clone() * 2.0 + z.clone().sigmoid();

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
        let (difference, gradient) = finite_difference(&mut y, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn mul_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 10));
        let mut y = ParameterNode::new(random_matrix(10, 10));
        let z = x.clone() * y.clone();
        let mut z = z.clone() + z.clone();

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
        let (difference, gradient) = finite_difference(&mut y, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn div_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(1, 1));
        let y = ParameterNode::new(random_matrix(1, 1));
        let mut z = (x.clone() + x.clone()) / y.clone();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn vector_dot_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut y = ParameterNode::new(random_matrix(10, 5));
        let z = x.vector_dot(&y);
        let mut z = z.clone() + z.clone();

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);

        let (difference, gradient) = finite_difference(&mut y, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn dot_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut y = ParameterNode::new(random_matrix(5, 10));
        let mut z = (x.clone() + x.clone()).dot(&y);

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);

        let (difference, gradient) = finite_difference(&mut y, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn dot_accumulation_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut y = ParameterNode::new(random_matrix(5, 10));
        let z = x.clone().dot(&y);
        let mut v = z.clone() * z.clone();

        let (difference, gradient) = finite_difference(&mut x, &mut v);
        assert_close(&difference, &gradient, TOLERANCE);

        let (difference, gradient) = finite_difference(&mut y, &mut v);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn square_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut z = x.square();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn ln_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(2, 2));
        let mut z = (x.clone() + x.clone()).exp().ln();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn tanh_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(2, 2));
        let mut z = (x.clone() + x.clone()).tanh();

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn sum_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut z = (x.clone() + x.clone()).scalar_sum();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE * 2.0);
    }
    #[test]
    fn squared_sum_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut z = x.square().scalar_sum();

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn transpose_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut z = (x.clone() + x.clone()).t();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn exp_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut z = (x.clone() + x.clone()).exp();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn dot_square_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let y = ParameterNode::new(random_matrix(10, 5));
        let mut z = x.vector_dot(&y).square();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn sigmoid_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let z = (x.clone() + x.clone()).sigmoid();
        let mut z = z.clone() + z.clone();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn relu_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let z = (x.clone() + x.clone()).relu();
        let mut z = z * 3.0;

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn neg_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut z = -(x.clone() + x.clone());

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn softmax_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(1, 10));
        let mut z = (x.clone() + x.clone()).softmax();

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn log_softmax_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(1, 10));
        let mut z = (x.clone() + x.clone()).log_softmax();
        let v = (x.clone() + x.clone()).softmax().ln();

        assert_close(v.value().deref(), z.value().deref(), TOLERANCE);

        let (finite_difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn sparse_categorical_cross_entropy_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(1, 10));
        let z = x.clone() + x.clone();
        let idx = IndexInputNode::new(&vec![0][..]);
        let mut loss = nn::losses::sparse_categorical_crossentropy(&z, &idx);

        let (finite_difference, gradient) = finite_difference(&mut x, &mut loss);
        assert_close(&finite_difference, &gradient, TOLERANCE);
    }
    #[test]
    fn rowwise_stack_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut y = ParameterNode::new(random_matrix(10, 5));
        //let v = x.clone() + y.clone();

        let z = x.stack(&y, ndarray::Axis(0));
        let mut z = z.clone().sigmoid() * z.clone().relu();

        assert_eq!(z.value().rows(), 20);
        assert_eq!(z.value().cols(), 5);

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);

        let (difference, gradient) = finite_difference(&mut y, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn columnwise_stack_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 5));
        let mut y = ParameterNode::new(random_matrix(10, 5));
        //let v = x.clone() + y.clone();

        let mut z = x.stack(&y, ndarray::Axis(1)).sigmoid();

        assert_eq!(z.value().rows(), 10);
        assert_eq!(z.value().cols(), 10);

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);

        let (difference, gradient) = finite_difference(&mut y, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn columnwise_view_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(10, 30));

        let x_0 = x.slice(s![.., 0..10]);
        let x_1 = x.slice(s![.., 10..20]);
        let x_2 = x.slice(s![.., 20..30]);

        assert_eq!(x_0.value().rows(), 10);
        assert_eq!(x_0.value().cols(), 10);
        assert_eq!(x_1.value().rows(), 10);
        assert_eq!(x_1.value().cols(), 10);
        assert_eq!(x_2.value().rows(), 10);
        assert_eq!(x_2.value().cols(), 10);

        let mut z = (x_0 + x_1 + x_2).sigmoid();

        let (difference, gradient) = finite_difference(&mut x, &mut z);
        assert_close(&difference, &gradient, TOLERANCE);
    }
    #[test]
    fn sparse_index_finite_difference() {
        let mut x = ParameterNode::new(random_matrix(100, 5));

        for _ in 0..10 {
            let idx_0 = IndexInputNode::new(&[random_index(10)]);
            let idx_1 = IndexInputNode::new(&[random_index(10)]);

            let mut z = (x.index(&idx_0).tanh() * x.index(&idx_1)).square();

            let (difference, gradient) = finite_difference(&mut x, &mut z);
            assert_close(&difference, &gradient, TOLERANCE);
        }
    }
    #[test]
    fn univariate_regression() {
        let slope = ParameterNode::new(random_matrix(1, 1));
        let intercept = ParameterNode::new(random_matrix(1, 1));

        let num_epochs = 200;

        let x = InputNode::new(random_matrix(1, 1));
        let y = InputNode::new(random_matrix(1, 1));

        let y_hat = slope.clone() * x.clone() + intercept.clone();
        let diff = y.clone() - y_hat.clone();
        let mut loss = diff.square();

        let optimizer = Adagrad::new().learning_rate(0.5);

        for _ in 0..num_epochs {
            let _x = arr2(&[[rand::thread_rng().gen()]]);
            let _y = 0.5 * &_x + 0.2;

            x.set_value(&_x);
            y.set_value(&_y);

            loss.forward();
            loss.backward(1.0);

            optimizer.step(loss.parameters());
        }

        println!(
            "Predicted: {} Loss: {} Slope {} Intercept {}",
            y_hat.value(),
            loss.value(),
            slope.value(),
            intercept.value()
        );

        assert!(loss.value().scalar_sum() < 1.0e-2);
    }

    #[test]
    fn multivariate_regression() {
        let slope = ParameterNode::new(random_matrix(1, 3));
        let intercept = ParameterNode::new(random_matrix(1, 1));

        let num_epochs = 200;

        let coefficients = arr2(&[[1.0], [2.0], [3.0]]);

        let x = InputNode::new(random_matrix(1, 3));
        let y = InputNode::new(random_matrix(1, 1));

        let y_hat = x.vector_dot(&slope) + intercept.clone();
        let diff = y.clone() - y_hat.clone();
        let mut loss = diff.square();

        let optimizer = SGD::new().learning_rate(0.1);

        for _ in 0..num_epochs {
            let _x = arr2(&[[
                rand::thread_rng().gen(),
                rand::thread_rng().gen(),
                rand::thread_rng().gen(),
            ]]);
            let _y = &_x.dot(&coefficients) + 5.0;

            x.set_value(&_x);
            y.set_value(&_y);

            loss.forward();
            loss.backward(1.0);

            optimizer.step(loss.parameters());
        }

        println!(
            "Predicted: {} Loss: {} Slope {} Intercept {}",
            y_hat.value(),
            loss.value(),
            slope.value(),
            intercept.value()
        );

        assert!(loss.value().scalar_sum() < 1.0e-1);
    }

    #[test]
    fn embedding_factorization() {
        let (rows, cols) = (10, 4);

        let true_u = random_matrix(rows, 10);
        let true_v = random_matrix(cols, 10);
        let x = true_u.dot(&true_v.t());

        let y = random_matrix(1, 1);
        let u_input = vec![0];
        let v_input = vec![0];

        let output = InputNode::new(y);

        let u_embedding = ParameterNode::new(random_matrix(rows, 10));
        let v_embedding = ParameterNode::new(random_matrix(cols, 10));

        let u_index = IndexInputNode::new(&u_input);
        let v_index = IndexInputNode::new(&v_input);

        let u_vec = u_embedding.index(&u_index);
        let v_vec = v_embedding.index(&v_index);

        let y_hat = u_vec.vector_dot(&v_vec);
        let mut loss = (output.clone() - y_hat.clone()).square();

        let num_epochs = 200;
        let optimizer = Adagrad::new().learning_rate(0.1);

        let mut loss_val = 0.0;

        for _ in 0..num_epochs {
            loss_val = 0.0;

            for row_idx in 0..rows {
                for col_idx in 0..cols {
                    u_index.set_value(row_idx);
                    v_index.set_value(col_idx);

                    output.set_value(x[(row_idx, col_idx)]);

                    loss.forward();
                    loss.backward(1.0);

                    loss_val += loss.value().scalar_sum();

                    optimizer.step(loss.parameters());
                }
            }

            println!("Loss {}", loss_val)
        }

        assert!(loss_val < 1e-2);
    }

    #[test]
    fn hogwild_embedding_factorization() {
        let (rows, cols) = (10, 4);

        let true_u = random_matrix(rows, 10);
        let true_v = random_matrix(cols, 10);
        let x = true_u.dot(&true_v.t());

        let u_input = vec![0];
        let v_input = vec![0];

        let u_parameters = Arc::new(HogwildParameter::new(random_matrix(rows, 10)));
        let v_parameters = Arc::new(HogwildParameter::new(random_matrix(cols, 10)));

        let losses: Vec<f64> = (0..rayon::current_num_threads())
            .into_par_iter()
            .map(|_| {
                let u_embedding = ParameterNode::shared(u_parameters.clone());
                let v_embedding = ParameterNode::shared(v_parameters.clone());

                let u_index = IndexInputNode::new(&u_input);
                let v_index = IndexInputNode::new(&v_input);
                let output = InputNode::new(random_matrix(1, 1));

                let u_vec = u_embedding.index(&u_index);
                let v_vec = v_embedding.index(&v_index);

                let y_hat = u_vec.vector_dot(&v_vec);
                let mut loss = (output.clone() - y_hat.clone()).square();

                let num_epochs = 100;

                let optimizer = SGD::new();

                let mut loss_val = 0.0;

                for _ in 0..num_epochs {
                    loss_val = 0.0;

                    for row_idx in 0..rows {
                        for col_idx in 0..cols {
                            u_index.set_value(row_idx);
                            v_index.set_value(col_idx);

                            output.set_value(x[(row_idx, col_idx)]);

                            loss.forward();
                            loss.backward(1.0);

                            loss_val += loss.value().scalar_sum();

                            optimizer.step(loss.parameters());
                        }
                    }
                }

                println!("Loss val {}", loss_val);

                loss_val
            })
            .collect();

        let sum_loss: f64 = losses.iter().sum();

        assert!(sum_loss / (losses.len() as f64) < 1e-3);
    }

    #[test]
    fn synchronized_embedding_factorization() {
        let (rows, cols) = (10, 4);

        let true_u = random_matrix(rows, 10);
        let true_v = random_matrix(cols, 10);
        let x = true_u.dot(&true_v.t());

        let u_input = vec![0];
        let v_input = vec![0];

        let u_parameters = Arc::new(HogwildParameter::new(random_matrix(rows, 10)));
        let v_parameters = Arc::new(HogwildParameter::new(random_matrix(cols, 10)));

        let optimizer = SGD::new();

        let losses: Vec<f64> = optimizer
            .synchronized(rayon::current_num_threads())
            .into_par_iter()
            .map(|optimizer| {
                let u_embedding = ParameterNode::shared(u_parameters.clone());
                let v_embedding = ParameterNode::shared(v_parameters.clone());

                let u_index = IndexInputNode::new(&u_input);
                let v_index = IndexInputNode::new(&v_input);
                let output = InputNode::new(random_matrix(1, 1));

                let u_vec = u_embedding.index(&u_index);
                let v_vec = v_embedding.index(&v_index);

                let y_hat = u_vec.vector_dot(&v_vec);
                let mut loss = (output.clone() - y_hat.clone()).square();

                let num_epochs = 100;

                let mut loss_val = 0.0;

                for _ in 0..num_epochs {
                    loss_val = 0.0;

                    for row_idx in 0..rows {
                        for col_idx in 0..cols {
                            u_index.set_value(row_idx);
                            v_index.set_value(col_idx);

                            output.set_value(x[(row_idx, col_idx)]);

                            loss.forward();
                            loss.backward(1.0);

                            loss_val += loss.value().scalar_sum();

                            optimizer.step(loss.parameters());
                        }
                    }
                }

                println!("Loss val {}", loss_val);

                loss_val
            })
            .collect();

        let sum_loss: f64 = losses.iter().sum();

        assert!(sum_loss / (losses.len() as f64) < 1e-3);
    }

}
*/