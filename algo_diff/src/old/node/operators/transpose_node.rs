use crate::maths::{Matrix, ArraySliceOps};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, BackwardAction, Bor};



#[derive(Debug)]
pub struct TransposeNode<OP> {
    value: RefCell<Matrix>,
    gradient: RefCell<Matrix>,
    operand: Rc<OP>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<OP> TransposeNode<OP>
where
    OP: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let needs_gradient = operand.needs_gradient();
        let mut value = Matrix::zeros((operand.value().ncols(), operand.value().nrows()));
        value.assign(&operand.value().t());
        let value = RefCell::new(value);
        let gradient = RefCell::new(operand.value().deref() * 0.0);

        TransposeNode {
            value: value,
            gradient: gradient,
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> Node for TransposeNode<OP>
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
        self.value.borrow_mut().assign(&self.operand.value().t());
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => {
                self.gradient.borrow_mut().assign(&gradient.t());
            }
            BackwardAction::Increment => {
                self.gradient.borrow_mut().slice_add_assign(&gradient.t());
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.gradient.borrow());
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
