use crate::maths::Matrix;
use std::cell::{Ref, RefCell};
use std::rc::Rc;
use super::{Node, Bor};
use super::Variable;



/// Input node for the graph.
#[derive(Debug)]
pub struct InputNode {
    pub(crate) value: RefCell<Matrix>,
}

impl InputNode {
    /// Create a new input node with a given value. This fixes the shape
    /// of the node in the graph.
    pub fn new(value: Matrix) -> Variable<Self> {
        Variable::new(
            Rc::new(InputNode {
                value: RefCell::new(value),
            }),
            Vec::new(),
        )
    }
}

impl Node for InputNode {
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {}
    fn backward(&self, _input_node: &Ref<Self::InputGradient>) {}
    fn value(&self) -> Bor<Self::Value> {
        Bor::RefGuard(self.value.borrow())
    }
    fn needs_gradient(&self) -> bool {
        false
    }
    fn clear(&self) {}
}
