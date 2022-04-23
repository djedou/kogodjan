use crate::maths::{Matrix, map_inplace_assign_binary, map_assign_binary, map_assign, sigmoid};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, BackwardAction, Bor};


#[derive(Debug)]
pub struct SigmoidNode<T> {
    value: RefCell<Matrix>,
    operand_gradient: RefCell<Matrix>,
    operand: Rc<T>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<T> SigmoidNode<T>
where
    T: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<T>) -> Self {
        let value = operand.value().deref().map(|&x| sigmoid(x));
        let gradient = &value * 0.0;
        let needs_gradient = operand.needs_gradient();

        SigmoidNode {
            value: RefCell::new(value),
            operand_gradient: RefCell::new(gradient),
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<T> Node for SigmoidNode<T>
where
    T: Node<Value = Matrix, InputGradient = Matrix>,
{
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {
        if self.counter.forward() == ForwardAction::Cached {
            return;
        }

        self.operand.forward();

        {
            let mut dest = self.value.borrow_mut();

            map_assign(&mut dest, self.operand.value().deref(), |x| {
                sigmoid(x)
            });
        }
    }

    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => {
                let mut operand_gradient = self.operand_gradient.borrow_mut();

                map_assign_binary(
                    &mut operand_gradient,
                    self.value.borrow().deref(),
                    gradient,
                    |sigmoid, grad| grad * sigmoid * (1.0 - sigmoid),
                );
            }
            BackwardAction::Increment => {
                let mut operand_gradient = self.operand_gradient.borrow_mut();

                map_inplace_assign_binary(
                    &mut operand_gradient,
                    self.value.borrow().deref(),
                    gradient,
                    |dest, sigmoid, grad| *dest += grad * sigmoid * (1.0 - sigmoid),
                );
            }
        }

        if self.counter.recurse_backward() {
            self.operand.backward(&self.operand_gradient.borrow())
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
