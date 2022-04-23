use crate::maths::Matrix;
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, BackwardAction, Bor};
use itertools::izip;


#[derive(Debug)]
pub struct SquareNode<OP> {
    value: RefCell<Matrix>,
    operand_gradient: RefCell<Matrix>,
    operand: Rc<OP>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<OP> SquareNode<OP>
where
    OP: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let value = operand.value().map(|x| x.powi(2));
        let gradient = &value * 0.0;
        let needs_gradient = operand.needs_gradient();

        SquareNode {
            value: RefCell::new(value),
            operand_gradient: RefCell::new(gradient),
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> Node for SquareNode<OP>
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

        dest.assign(self.operand.value().deref());
        dest.map_inplace(|x| *x = x.powi(2));
    }

    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => for (dest, operand_val, grad_val) in izip!(
                self.operand_gradient.borrow_mut().iter_mut(),
                self.operand.value().iter(),
                gradient.iter()
            ) {
                *dest = operand_val * 2.0 * (*grad_val);
            },
            BackwardAction::Increment => for (dest, operand_val, grad_val) in izip!(
                self.operand_gradient.borrow_mut().iter_mut(),
                self.operand.value().iter(),
                gradient.iter()
            ) {
                *dest += operand_val * 2.0 * (*grad_val);
            },
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
