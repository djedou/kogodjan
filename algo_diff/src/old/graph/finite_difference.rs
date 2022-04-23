use super::{DataInput};
use crate::maths::{Matrix};
use crate::node::{ParameterNode, Node, Variable};


/// Compute finite difference gradient estimates of the output variable
/// with respect to the input. Use to verify correctness of gradient
/// computations.
pub fn finite_difference<T>(
    input: &mut Variable<ParameterNode>,
    output: &mut Variable<T>,
) -> (Matrix, Matrix)
where
    T: Node<Value = Matrix, InputGradient = Matrix>,
{
    let delta_x = 1e-4;

    let initial_input = { input.value().clone() };
    let mut central_difference = &initial_input * 0.0;

    for (idx, diff) in central_difference.indexed_iter_mut() {
        let positive_difference = {
            output.zero_gradient();
            let mut changed_input = initial_input.clone();
            changed_input[idx] += 0.5 * delta_x;
            input.set_value(&changed_input);
            output.forward();
            output.backward(1.0);
            output.value().clone()
        };

        let negative_difference = {
            output.zero_gradient();
            let mut changed_input = initial_input.clone();
            changed_input[idx] -= 0.5 * delta_x;
            input.set_value(&changed_input);
            output.forward();
            output.backward(1.0);
            output.value().clone()
        };

        let central_difference = positive_difference - negative_difference;

        *diff = central_difference.sum() / delta_x;
    }

    let gradient = {
        output.zero_gradient();
        input.set_value(&initial_input);
        output.forward();
        output.backward(1.0);

        input.gradient()
    };

    output.zero_gradient();

    (central_difference, gradient)
}
