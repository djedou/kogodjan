use crate::maths::{Matrix, mul, increment_mul};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use super::{PassCounter, Node, ForwardAction, BackwardAction, Bor};


#[derive(Debug)]
pub struct MulNode<LHS, RHS> {
    value: RefCell<Matrix>,
    lhs_gradient: RefCell<Matrix>,
    rhs_gradient: RefCell<Matrix>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<LHS, RHS> MulNode<LHS, RHS>
where
    LHS: Node<Value = Matrix>,
    RHS: Node<Value = Matrix>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let needs_gradient = lhs.needs_gradient() || rhs.needs_gradient();
        let value = lhs.value().deref() * rhs.value().deref();

        let lhs_gradient = &value * 0.0;
        let rhs_gradient = &value * 0.0;

        MulNode {
            value: RefCell::new(value),
            lhs_gradient: RefCell::new(lhs_gradient),
            rhs_gradient: RefCell::new(rhs_gradient),
            lhs: lhs,
            rhs: rhs,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Node for MulNode<LHS, RHS>
where
    LHS: Node<Value = Matrix, InputGradient = Matrix>,
    RHS: Node<Value = Matrix, InputGradient = Matrix>,
{
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {
        if self.counter.forward() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();
        self.rhs.forward();

        let mut dest = self.value.borrow_mut();

        mul(
            self.lhs.value().deref(),
            self.rhs.value().deref(),
            &mut dest
        );
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => {
                let mut lhs_gradient = self.lhs_gradient.borrow_mut();

                mul(
                    self.rhs.value().deref(),
                    gradient.deref(),
                    &mut lhs_gradient
                );

                let mut rhs_gradient = self.rhs_gradient.borrow_mut();

                mul(
                    self.lhs.value().deref(),
                    gradient.deref(),
                    &mut rhs_gradient
                );
            }
            BackwardAction::Increment => {
                let mut lhs_gradient = self.lhs_gradient.borrow_mut();
                let mut rhs_gradient = self.rhs_gradient.borrow_mut();

                increment_mul(
                    self.rhs.value().deref(),
                    gradient.deref(),
                    &mut lhs_gradient
                );
                increment_mul(
                    self.lhs.value().deref(),
                    gradient.deref(),
                    &mut rhs_gradient
                );
            }
        }

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_gradient.borrow());
            self.rhs.backward(&self.rhs_gradient.borrow());
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
            self.lhs.clear();
            self.rhs.clear();
            self.counter.clear();
        }
    }
}
