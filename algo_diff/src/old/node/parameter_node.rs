use std::cell::{Ref, RefCell};
use std::rc::Rc;
use std::sync::Arc;
use super::Variable;
use crate::maths::Matrix;
use super::*;

/// Parameter node, holds the optimizable parameters of the model.
#[derive(Debug)]
pub struct ParameterNode {
    pub(crate) value: Arc<HogwildParameter>,
    pub(crate) gradient: RefCell<GradientAccumulator>,
}

impl ParameterNode {
    /// Create a parameter node that shares its parameter values
    /// with other parameter nodes via the `HogwildParameter` object.
    pub fn shared(value: Arc<HogwildParameter>) -> Variable<Self> {
        let shape = unsafe {
            // This method can be called in multiple threads, so borrowing
            // (even immutably) will read to borrow failures.
            (
                (*value.value.as_ptr()).nrows(),
                (*value.value.as_ptr()).ncols(),
            )
        };

        let node = Rc::new(ParameterNode {
            value: value,
            gradient: RefCell::new(GradientAccumulator::new(shape)),
        });
        let params = vec![Variable::new(Rc::clone(&node), Vec::new())];

        Variable::new(node, params)
    }
    /// Create a new parameter node. The parameters held by this node
    /// cannot be shared and optimized in parallel.
    pub fn new(value: Matrix) -> Variable<Self> {
        let shape = (value.nrows(), value.ncols());

        let node = Rc::new(ParameterNode {
            value: Arc::new(HogwildParameter::new(value)),
            gradient: RefCell::new(GradientAccumulator::new(shape)),
        });
        let params = vec![Variable::new(Rc::clone(&node), Vec::new())];

        Variable::new(node, params)
    }

    pub(crate) fn zero_gradient(&self) {
        self.gradient.borrow_mut().zero_gradient();
    }
}

impl Node for ParameterNode {
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {}
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        self.gradient.borrow_mut().accumulate_gradient(gradient);
    }
    fn value(&self) -> Bor<Self::Value> {
        Bor::Reference(unsafe { &*(self.value.value.as_ptr() as *const Matrix) })
    }
    fn needs_gradient(&self) -> bool {
        true
    }
    fn clear(&self) {}
}
