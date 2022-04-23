use super::Optimizer;
use crate::maths::{ArraySlice, map_add_assign_slice};
use crate::node::{Variable};
use crate::node::{ParameterNode, HogwildParameter, GradientAccumulator};
use std::ops::DerefMut;
use std::sync::Arc;
use serde_derive::{Serialize, Deserialize};

use ndarray::Axis;

/// Standard stochastic gradient descent optimizer with a fixed learning rate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGD {
    learning_rate: f64,
    clamp: Option<(f64, f64)>,
}

impl Default for SGD {
    fn default() -> Self {
        Self::new()
    }
}

impl SGD {
    /// Create a new optimizer instance with a given set of parameters.
    pub fn new() -> Self {
        SGD {
            learning_rate: 0.05,
            clamp: None,
        }
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the clamp bounds.
    pub fn clamp(mut self, min: f64, max: f64) -> Self {
        self.clamp = Some((min, max));
        self
    }

    fn inner_step<T: DerefMut<Target = GradientAccumulator>>(
        &self,
        param: &Arc<HogwildParameter>,
        mut sink: T,
    ) {
        let param_value = unsafe { param.value_mut() };
        let learning_rate = self.learning_rate;
        let sink = sink.deref_mut();

        if let Some((min, max)) = self.clamp {
            sink.clamp(min, max);
        }

        if sink.has_dense() {
            param_value.scaled_add(-self.learning_rate, sink.gradient());
        } else {
            for (row_idx, grad) in sink.sparse_iter() {
                let param_row = param_value.index_axis(Axis(0), row_idx);

                map_add_assign_slice(
                    &mut param_row.to_vec(),
                    grad.fast_slice(),
                    |x| -learning_rate * x,
                );
            }
        }
    }
}

impl Optimizer for SGD {
    fn step(&self, parameters: &[Variable<ParameterNode>]) {
        for parameter in parameters {
            self.inner_step(&parameter.node.value, parameter.node.gradient.borrow_mut());
            parameter.node.zero_gradient();
        }
    }
}
