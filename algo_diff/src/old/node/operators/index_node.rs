use crate::maths::{Matrix, ArraySliceOps};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, Bor, GradientSink};
use smallvec::SmallVec;
use ndarray::Axis;
use crate::node::{ParameterNode, IndexInputNode};


#[derive(Debug)]
pub struct IndexNode<OP> {
    value: RefCell<Matrix>,
    index_value: RefCell<SmallVec<[usize; 4]>>,
    operand_gradient: RefCell<Matrix>,
    index: Rc<IndexInputNode>,
    operand: Rc<OP>,
    needs_gradient: bool,
    counter: PassCounter,
}


impl<OP> IndexNode<OP>
where
    OP: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<OP>, index: Rc<IndexInputNode>) -> Self {
        let value = operand.value().select(Axis(0), &index.value()[..]);
        let grad = &value * 0.0;
        let idx_value = index.value().clone();
        let needs_gradient = operand.needs_gradient();

        IndexNode {
            value: RefCell::new(value),
            index_value: RefCell::new(idx_value),
            operand_gradient: RefCell::new(grad),
            index: index,
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl Node for IndexNode<ParameterNode> {
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {
        if self.counter.forward() == ForwardAction::Cached {
            return;
        }

        let operand_value = self.operand.value();

        let mut idx_value = self.index_value.borrow_mut();
        idx_value.clear();
        idx_value.extend_from_slice(&self.index.value()[..]);

        let mut arr_value = self.value.borrow_mut();

        debug_assert_eq!(
            arr_value.shape()[0],
            idx_value.len(),
            "Result of indexing operation must maintain consistent shape between iterations."
        );

        for (&idx, mut row) in idx_value.iter().zip(arr_value.rows_mut()) {
            let new_val = operand_value.index_axis(Axis(0), idx);

            row.slice_assign(&new_val);
        }
    }

    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        self.counter.backward();
        self.operand
            .gradient
            .borrow_mut()
            .accumulate_gradient((&self.index_value.borrow()[..], gradient.deref()));
        self.counter.recurse_backward();
    }

    fn value(&self) -> Bor<Self::Value> {
        Bor::RefGuard(self.value.borrow())
    }

    fn needs_gradient(&self) -> bool {
        self.needs_gradient
    }
    fn clear(&self) {
        if !self.counter.is_zero() {
            self.operand.clear();
            self.counter.clear();
        }
    }
}