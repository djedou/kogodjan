use crate::maths::{Matrix, map_assign, tanh};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, BackwardAction, Bor};
use itertools::izip;


#[derive(Debug)]
pub struct TanhNode<OP> {
    value: RefCell<Matrix>,
    operand_gradient: RefCell<Matrix>,
    operand: Rc<OP>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<OP> TanhNode<OP>
where
    OP: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let value = operand.value().map(|&x| tanh(x));
        let gradient = &value * 0.0;
        let needs_gradient = operand.needs_gradient();

        TanhNode {
            value: RefCell::new(value),
            operand_gradient: RefCell::new(gradient),
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> Node for TanhNode<OP>
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
        map_assign(&mut dest, self.operand.value().deref(), |x| {
            tanh(x)
        });
    }

    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => for (dest, value, grad_val) in izip!(
                self.operand_gradient.borrow_mut().as_slice_mut().unwrap(),
                self.value().as_slice().unwrap(),
                gradient.as_slice().unwrap()
            ) {
                *dest = grad_val * (1.0 - value.powi(2));
            },
            BackwardAction::Increment => for (dest, value, grad_val) in izip!(
                self.operand_gradient.borrow_mut().as_slice_mut().unwrap(),
                self.value().as_slice().unwrap(),
                gradient.as_slice().unwrap()
            ) {
                *dest += grad_val * (1.0 - value.powi(2));
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
