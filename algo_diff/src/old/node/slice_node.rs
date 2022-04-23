use crate::maths::{Matrix};
use std::cell::{Ref, RefCell};
use std::ops::{Deref, AddAssign};
use std::rc::Rc;
use super::{PassCounter, Node, ForwardAction, BackwardAction, Bor};


#[derive(Debug)]
pub struct SliceNode<LHS> {
    slice: ndarray::SliceInfo<[ndarray::SliceInfoElem; 2], ndarray::Ix2, ndarray::Ix2>,
    value: RefCell<Matrix>,
    lhs_gradient: RefCell<Matrix>,
    lhs: Rc<LHS>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<LHS> SliceNode<LHS>
where
    LHS: Node<Value = Matrix>,
{
    pub fn new(
        lhs: Rc<LHS>,
        slice: &ndarray::SliceInfo<[ndarray::SliceInfoElem; 2], ndarray::Ix2, ndarray::Ix2>,
    ) -> Self {
        let needs_gradient = lhs.needs_gradient();

        let value = {
            let val = lhs.value();
            let sliced = val.slice(slice);
            let mut value = Matrix::zeros((sliced.nrows(), sliced.ncols()));
            value.assign(&sliced);

            value
        };

        let lhs_gradient = lhs.value().deref() * 0.0;

        SliceNode {
            slice: *slice,
            value: RefCell::new(value),
            lhs_gradient: RefCell::new(lhs_gradient),
            lhs: lhs,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<LHS> Node for SliceNode<LHS>
where
    LHS: Node<Value = Matrix, InputGradient = Matrix>,
{
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {
        if self.counter.forward() == ForwardAction::Cached {
            return;
        }

        self.lhs.forward();

        let lhs_value = self.lhs.value();
        let mut self_value = self.value.borrow_mut();
        self_value.assign(&lhs_value.slice(&self.slice));
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        match self.counter.backward() {
            BackwardAction::Set => {
                self.lhs_gradient
                    .borrow_mut()
                    .slice_mut(&self.slice)
                    .assign(gradient.deref());
            }
            BackwardAction::Increment => {
                self.lhs_gradient
                    .borrow_mut()
                    .slice_mut(&self.slice)
                    .add_assign(gradient.deref());
            }
        }

        if self.counter.recurse_backward() {
            self.lhs.backward(&self.lhs_gradient.borrow());
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
            self.counter.clear();
        }
    }
}
