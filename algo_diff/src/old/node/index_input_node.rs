use crate::maths::{Matrix};
use std::cell::{Ref, RefCell};
use std::rc::Rc;
use super::{Node, Bor};
use smallvec::SmallVec;
use super::Variable;


/// An input node for integer indices into `ParameterNode`s, used
/// for implementing indexable embedding layers.
#[derive(Debug)]
pub struct IndexInputNode {
    pub(crate) value: RefCell<SmallVec<[usize; 4]>>,
}

impl IndexInputNode {
    /// Create a new index input node.
    pub fn new(value: &[usize]) -> Variable<Self> {
        Variable::new(
            Rc::new(IndexInputNode {
                value: RefCell::new(SmallVec::from(value)),
            }),
            Vec::new(),
        )
    }
}

impl Node for IndexInputNode {
    type Value = SmallVec<[usize; 4]>;
    type InputGradient = Matrix;
    fn forward(&self) {}
    fn backward(&self, _: &Ref<Self::InputGradient>) {}
    fn value(&self) -> Bor<Self::Value> {
        Bor::RefGuard(self.value.borrow())
    }
    fn needs_gradient(&self) -> bool {
        false
    }
    fn clear(&self) {}
}
