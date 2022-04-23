use crate::maths::{Matrix, simd_scaled_assign, sub, ArraySliceOps};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use super::{PassCounter, Node, ForwardAction, BackwardAction, Bor};


#[derive(Debug)]
pub struct SubNode<LHS, RHS>
where
    LHS: Node<Value = Matrix, InputGradient = Matrix>,
    RHS: Node<Value = Matrix, InputGradient = Matrix>,
{
    value: RefCell<Matrix>,
    lhs_gradient: RefCell<Matrix>,
    rhs_gradient: RefCell<Matrix>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<LHS, RHS> SubNode<LHS, RHS>
where
    LHS: Node<Value = Matrix, InputGradient = Matrix>,
    RHS: Node<Value = Matrix, InputGradient = Matrix>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let needs_gradient = lhs.needs_gradient() || rhs.needs_gradient();
        let value = lhs.value().deref() - rhs.value().deref();

        let rhs_gradient = rhs.value().deref() * 0.0;
        let lhs_gradient = lhs.value().deref() * 0.0;

        SubNode {
            value: RefCell::new(value),
            rhs_gradient: RefCell::new(rhs_gradient),
            lhs_gradient: RefCell::new(lhs_gradient),
            lhs: lhs,
            rhs: rhs,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Node for SubNode<LHS, RHS>
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

        sub(
            self.lhs.value().deref(),
            self.rhs.value().deref(),
            &mut dest
        );
    }

    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => {
                let mut rhs_gradient = self.rhs_gradient.borrow_mut();

                simd_scaled_assign(
                    rhs_gradient.as_slice_mut().unwrap(),
                    gradient.as_slice().unwrap(),
                    -1.0,
                );

                let mut lhs_gradient = self.lhs_gradient.borrow_mut();

                simd_scaled_assign(
                    lhs_gradient.as_slice_mut().unwrap(),
                    gradient.as_slice().unwrap(),
                    1.0,
                );
            }
            BackwardAction::Increment => {
                let mut rhs_gradient = self.rhs_gradient.borrow_mut();
                rhs_gradient.slice_sub_assign(gradient.deref());

                let mut lhs_gradient = self.lhs_gradient.borrow_mut();
                lhs_gradient.slice_add_assign(gradient.deref());
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
