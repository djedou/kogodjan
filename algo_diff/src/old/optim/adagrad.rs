use super::Optimizer;
use crate::maths::{ArraySlice, ArraySliceMut, pow2, ArraySliceOps};
use crate::node::{HogwildParameter, ParameterNode, GradientAccumulator};
use crate::graph::Variable;
use std::ops::DerefMut;
use std::sync::Arc;
use serde_derive::{Serialize, Deserialize};
use ndarray::Axis;
use itertools::izip;



/// Adagrad optimizer, scaled the learning rate by the inverse of previously
/// accumulated gradients.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adagrad {
    learning_rate: f64,
    l2: f64,
    clamp: Option<(f64, f64)>,
    eps: f64,
}

impl Default for Adagrad {
    fn default() -> Self {
        Self::new()
    }
}

impl Adagrad {
    /// Create a new optimizer instance with a given set of parameters.
    pub fn new() -> Self {
        Adagrad {
            learning_rate: 0.05,
            l2: 0.0,
            clamp: None,
            eps: 1e-10,
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

    /// Set the L2 penalty.
    pub fn l2_penalty(mut self, l2_penalty: f64) -> Self {
        self.l2 = l2_penalty;
        self
    }

    fn inner_step<T: DerefMut<Target = GradientAccumulator>>(
        &self,
        param: &Arc<HogwildParameter>,
        mut sink: T,
    ) {
        let learning_rate = self.learning_rate;

        if let Some((min, max)) = self.clamp {
            sink.clamp(min, max);
        }

        let param_value = unsafe { param.value_mut() };
        let squared_gradient = unsafe { param.squared_gradient_mut() };

        if sink.has_dense() {
            for (value, &gradient, squared_gradient) in izip!(
                param_value.fast_slice_mut(),
                sink.gradient().fast_slice(),
                squared_gradient.fast_slice_mut()
            ) {
                let gradient = gradient + *value * self.l2;
                *squared_gradient += pow2(gradient);
                *value -= learning_rate / (self.eps + squared_gradient.sqrt()) * gradient;
            }
        } else {
            for (row_idx, grad) in sink.sparse_iter() {
                let mut param_row = param_value.index_axis(Axis(0), row_idx);
                let mut squared_row = squared_gradient.index_axis(Axis(0), row_idx);

                for (value, &gradient, squared_gradient) in izip!(
                    param_row.fast_slice_mut(),
                    grad.iter(),
                    squared_row.fast_slice_mut()
                ) {
                    let gradient = gradient + *value * self.l2;
                    *squared_gradient += pow2(gradient);
                    *value -= learning_rate / (self.eps + squared_gradient.sqrt()) * gradient;
                }
            }
        }
    }
}

impl Optimizer for Adagrad {
    /// Perform a single SGD step.
    fn step(&self, parameters: &[Variable<ParameterNode>]) {
        for parameter in parameters {
            self.inner_step(&parameter.node.value, parameter.node.gradient.borrow_mut());
            parameter.node.zero_gradient();
        }
    }
}