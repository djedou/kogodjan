use crate::maths::Matrix;
use std::cell::{Ref, RefCell};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use super::{PassCounter, Node, ForwardAction, Bor};
use super::{row_wise_stack, column_wise_stack, row_wise_stack_gradient, column_wise_stack_gradient};




#[derive(Debug)]
pub struct ConcatenateNode<LHS, RHS> {
    axis: ndarray::Axis,
    value: RefCell<Matrix>,
    lhs_gradient: RefCell<Matrix>,
    rhs_gradient: RefCell<Matrix>,
    lhs: Rc<LHS>,
    rhs: Rc<RHS>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<LHS, RHS> ConcatenateNode<LHS, RHS>
where
    LHS: Node<Value = Matrix>,
    RHS: Node<Value = Matrix>,
{
    pub fn new(lhs: Rc<LHS>, rhs: Rc<RHS>, axis: ndarray::Axis) -> Self {
        let needs_gradient = lhs.needs_gradient() || rhs.needs_gradient();

        let value = ndarray::concatenate(
            axis,
            &[lhs.value().deref().view(), rhs.value().deref().view()]
        ).expect("Unable to concatenate arrays.");

        let lhs_gradient = lhs.value().deref() * 0.0;
        let rhs_gradient = rhs.value().deref() * 0.0;

        ConcatenateNode {
            axis: axis,
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

impl<LHS, RHS> Node for ConcatenateNode<LHS, RHS>
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

        let mut self_value = self.value.borrow_mut();

        match self.axis {
            // Vertically
            ndarray::Axis(0) => {
                row_wise_stack(self_value.deref_mut(), lhs_value.deref(), rhs_value.deref())
            }
            // Horizontally
            ndarray::Axis(1) => {
                column_wise_stack(self_value.deref_mut(), lhs_value.deref(), rhs_value.deref())
            }
            // Not allowed
            _ => panic!("Stacking tensors not allowed."),
        }
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        {
            let mut lhs_grad = self.lhs_gradient.borrow_mut();
            let mut rhs_grad = self.rhs_gradient.borrow_mut();

            match self.axis {
                ndarray::Axis(0) => row_wise_stack_gradient(
                    gradient,
                    lhs_grad.deref_mut(),
                    rhs_grad.deref_mut(),
                    &self.counter.backward(),
                ),
                ndarray::Axis(1) => column_wise_stack_gradient(
                    gradient,
                    lhs_grad.deref_mut(),
                    rhs_grad.deref_mut(),
                    &self.counter.backward(),
                ),
                _ => panic!("Stacking tensors not allowed."),
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
