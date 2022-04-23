use crate::maths::{Matrix, mat_mul, ArraySliceOps};
use std::cell::{Ref, RefCell};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use super::{PassCounter, Node, ForwardAction, Bor, BackwardAction};


#[derive(Debug)]
pub struct DotNode<LHS, RHS> {
    value: RefCell<Matrix>,
    gradient: RefCell<Matrix>,
    lhs_gradient: RefCell<Matrix>,
    rhs_gradient: RefCell<Matrix>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<LHS, RHS> DotNode<LHS, RHS>
where
    LHS: Node<Value = Matrix>,
    RHS: Node<Value = Matrix>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let needs_gradient = lhs.needs_gradient() || rhs.needs_gradient();
        let value = lhs.value().dot(rhs.value().deref());
        let gradient = &value * 0.0;

        let lhs_gradient = lhs.value().deref() * 0.0;
        let rhs_gradient = rhs.value().deref() * 0.0;

        DotNode {
            value: RefCell::new(value),
            gradient: RefCell::new(gradient),
            lhs_gradient: RefCell::new(lhs_gradient),
            rhs_gradient: RefCell::new(rhs_gradient),
            lhs: lhs,
            rhs: rhs,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Node for DotNode<LHS, RHS>
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

        mat_mul(
            1.0,
            self.lhs.value().deref(),
            self.rhs.value().deref(),
            0.0,
            self.value.borrow_mut().deref_mut(),
        );
    }

    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => {
                self.gradient.borrow_mut().slice_assign(gradient.deref());
            }
            BackwardAction::Increment => {
                self.gradient
                    .borrow_mut()
                    .slice_add_assign(gradient.deref());
            }
        }

        if self.counter.recurse_backward() {
            {
                let rhs_value = self.rhs.value();
                let lhs_value = self.lhs.value();

                let gradient = self.gradient.borrow();

                let mut lhs_gradient = self.lhs_gradient.borrow_mut();
                let mut rhs_gradient = self.rhs_gradient.borrow_mut();

                mat_mul(
                    1.0,
                    gradient.deref(),
                    &rhs_value.t(),
                    0.0,
                    &mut lhs_gradient,
                );
                mat_mul(
                    1.0,
                    &lhs_value.t(),
                    gradient.deref(),
                    0.0,
                    &mut rhs_gradient,
                );
            }

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
