use crate::maths::{Matrix, ArraySliceOps};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, BackwardAction, Bor};


#[derive(Debug)]
pub struct SumNode<OP> {
    value: RefCell<Matrix>,
    operand_gradient: RefCell<Matrix>,
    operand: Rc<OP>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<OP> SumNode<OP>
where
    OP: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let value = {
            let mut value = Matrix::zeros((1, 1));
            value.fill(operand.value().sum());
            value
        };

        let gradient = operand.value().deref() * 0.0;
        let needs_gradient = operand.needs_gradient();

        SumNode {
            value: RefCell::new(value),
            operand_gradient: RefCell::new(gradient),
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> Node for SumNode<OP>
where
    OP: Node<Value = Matrix, InputGradient = Matrix>,
{
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {
        if self.counter.forward() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        let mut dest = self.value.borrow_mut();
        dest[(0, 0)] = self.operand.value().sum();
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        debug_assert!(gradient.len() == 1, "Input gradient must be a scalar.");

        match self.counter.backward() {
            BackwardAction::Set => {
                self.operand_gradient.borrow_mut().fill(gradient[(0, 0)]);
            }
            BackwardAction::Increment => {
                self.operand_gradient
                    .borrow_mut()
                    .slice_add_assign(gradient[(0, 0)]);
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.operand_gradient.borrow());
        }
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
