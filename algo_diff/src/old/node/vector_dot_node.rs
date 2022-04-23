use crate::maths::{Matrix, simd_scaled_add, simd_dot, simd_scaled_assign};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use super::{PassCounter, Node, ForwardAction, BackwardAction, Bor};
use itertools::izip;


#[derive(Debug)]
pub struct VectorDotNode<LHS, RHS> {
    value: RefCell<Matrix>,
    lhs_gradient: RefCell<Matrix>,
    rhs_gradient: RefCell<Matrix>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<LHS, RHS> VectorDotNode<LHS, RHS>
where
    LHS: Node<Value = Matrix, InputGradient = Matrix>,
    RHS: Node<Value = Matrix, InputGradient = Matrix>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>) -> Self {
        let (value, lhs_gradient, rhs_gradient, needs_gradient) = {
            let lhs_value = lhs.value();
            let rhs_value = rhs.value();

            let needs_gradient = lhs.needs_gradient() || rhs.needs_gradient();

            assert_eq!(
                lhs_value.shape(),
                rhs_value.shape(),
                "LHS and RHS must be the same shape for vector dot product."
            );

            let mut value = Matrix::zeros((lhs_value.shape()[0], 1));

            for (result, lhs, rhs) in izip!(
                value.as_slice_mut().unwrap(),
                lhs_value
                    .rows()
                    .into_iter()
                    .map(|x| x.to_slice().unwrap()),
                rhs_value
                    .rows()
                    .into_iter()
                    .map(|x| x.to_slice().unwrap())
            ) {
                *result = simd_dot(lhs, rhs);
            }

            let lhs_gradient = lhs_value.deref() * 0.0;
            let rhs_gradient = rhs_value.deref() * 0.0;

            (value, lhs_gradient, rhs_gradient, needs_gradient)
        };

        VectorDotNode {
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

impl<LHS, RHS> Node for VectorDotNode<LHS, RHS>
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

        for (result, lhs, rhs) in izip!(
            self.value.borrow_mut().as_slice_mut().unwrap(),
            lhs_value
                .rows()
                .into_iter()
                .map(|x| x.to_slice().unwrap()),
            rhs_value
                .rows()
                .into_iter()
                .map(|x| x.to_slice().unwrap())
        ) {
            *result = simd_dot(lhs, rhs);
        }
    }

    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        let lhs_value = self.lhs.value();
        let rhs_value = self.rhs.value();

        match self.counter.backward() {
            BackwardAction::Set => {
                let mut lhs_grad = self.lhs_gradient.borrow_mut();
                let mut rhs_grad = self.rhs_gradient.borrow_mut();

                for (backward_row, rhs_row, &gradient) in izip!(
                    lhs_grad
                        .rows_mut()
                        .into_iter()
                        .map(|x| x.into_slice().unwrap()),
                    rhs_value
                        .rows()
                        .into_iter()
                        .map(|x| x.to_slice().unwrap()),
                    gradient.as_slice().unwrap()
                ) {
                    simd_scaled_assign(backward_row, rhs_row, gradient)
                }
                for (backward_row, lhs_row, &gradient) in izip!(
                    rhs_grad
                        .rows_mut()
                        .into_iter()
                        .map(|x| x.into_slice().unwrap()),
                    lhs_value
                        .rows()
                        .into_iter()
                        .map(|x| x.to_slice().unwrap()),
                    gradient.as_slice().unwrap()
                ) {
                    simd_scaled_assign(backward_row, lhs_row, gradient)
                }
            }
            BackwardAction::Increment => {
                let mut lhs_grad = self.lhs_gradient.borrow_mut();
                let mut rhs_grad = self.rhs_gradient.borrow_mut();

                for (backward_row, rhs_row, &gradient) in izip!(
                    lhs_grad
                        .rows_mut()
                        .into_iter()
                        .map(|x| x.into_slice().unwrap()),
                    rhs_value
                        .rows()
                        .into_iter()
                        .map(|x| x.to_slice().unwrap()),
                    gradient.as_slice().unwrap()
                ) {
                    simd_scaled_add(backward_row, rhs_row, gradient)
                }
                for (backward_row, lhs_row, &gradient) in izip!(
                    rhs_grad
                        .rows_mut()
                        .into_iter()
                        .map(|x| x.into_slice().unwrap()),
                    lhs_value
                        .rows()
                        .into_iter()
                        .map(|x| x.to_slice().unwrap()),
                    gradient.as_slice().unwrap()
                ) {
                    simd_scaled_add(backward_row, lhs_row, gradient)
                }
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
