use crate::maths::{Matrix, ArraySlice, ArraySliceMut, ArraySliceOps};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use super::{PassCounter, Node, ForwardAction, BackwardAction, Bor};
use itertools::izip;



#[derive(Debug)]
pub struct AddNode<LHS, RHS> {
    value: RefCell<Matrix>,
    gradient: RefCell<Matrix>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<LHS, RHS> AddNode<LHS, RHS>
where
    LHS: Node<Value = Matrix>,
    RHS: Node<Value = Matrix>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let needs_gradient = lhs.needs_gradient() || rhs.needs_gradient();
        let value = lhs.value().deref() + rhs.value().deref();
        let gradient = rhs.value().deref() * 0.0;

        AddNode {
            value: RefCell::new(value),
            gradient: RefCell::new(gradient),
            lhs: lhs,
            rhs: rhs,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS, RHS> Node for AddNode<LHS, RHS>
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

        let lhs_value = self.lhs.value();
        let rhs_value = self.rhs.value();

        debug_assert_eq!(
            lhs_value.shape(),
            self.value().shape(),
            "LHS operand changed shape."
        );
        debug_assert_eq!(
            rhs_value.shape(),
            self.value().shape(),
            "RHS operand changed shape."
        );

        let mut self_value = self.value.borrow_mut();

        for (v, &lhs, &rhs) in izip!(
            self_value.fast_slice_mut(),
            lhs_value.fast_slice(),
            rhs_value.fast_slice()
        ) {
            *v = lhs + rhs;
        }
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => {
                let mut operand_gradient = self.gradient.borrow_mut();
                operand_gradient.slice_assign(gradient.deref());
            }
            BackwardAction::Increment => {
                let mut operand_gradient = self.gradient.borrow_mut();
                operand_gradient.slice_add_assign(gradient.deref());
            }
        }

        if self.counter.recurse_backward() {
            let gradient = self.gradient.borrow();
            self.lhs.backward(&gradient);
            self.rhs.backward(&gradient);
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

