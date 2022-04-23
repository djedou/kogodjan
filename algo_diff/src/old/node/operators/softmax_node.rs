use crate::maths::{Matrix, mat_mul, exp, ArraySlice, ArraySliceOps};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, BackwardAction, Bor};


#[derive(Debug)]
pub struct SoftmaxNode<OP> {
    value: RefCell<Matrix>,
    jacobian: RefCell<Matrix>,
    operand_gradient: RefCell<Matrix>,
    operand: Rc<OP>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<OP> SoftmaxNode<OP>
where
    OP: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let value = {
            let max = operand
                .value()
                .deref()
                .as_slice()
                .unwrap()
                .iter()
                .fold(std::f64::MIN, |x, y| x.max(*y));
            let numerator = operand.value().map(|x| exp(*x - max));
            let denominator = numerator.sum();

            numerator / denominator
        };

        let gradient = &value * 0.0;
        let needs_gradient = operand.needs_gradient();
        let dim = value.shape()[1];

        SoftmaxNode {
            value: RefCell::new(value),
            jacobian: RefCell::new(ndarray::Array2::zeros((dim, dim))),
            operand_gradient: RefCell::new(gradient),
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }
}

impl<OP> Node for SoftmaxNode<OP>
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
        dest.slice_assign(self.operand.value().deref());

        let max = self
            .operand
            .value()
            .fast_slice()
            .iter()
            .fold(std::f64::MIN, |x, y| x.max(*y));
        dest.map_inplace(|x| *x = exp(*x - max));
        let denominator = dest.sum();
        dest.map_inplace(|x| *x /= denominator);
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        // TODO: accumulate gradients
        let value = self.value.borrow();
        let mut jacobian = self.jacobian.borrow_mut();

        let beta = match self.counter.backward() {
            BackwardAction::Set => 0.0,
            BackwardAction::Increment => 1.0,
        };

        for (row_idx, (mut row, row_val)) in jacobian
            .rows_mut()
            .into_iter()
            .zip(value.iter())
            .enumerate()
        {
            for (col_idx, (grad, col_val)) in row
                .as_slice_mut()
                .unwrap()
                .iter_mut()
                .zip(value.as_slice().unwrap())
                .enumerate()
            {
                if row_idx == col_idx {
                    *grad = row_val * (1.0 - (*col_val));
                } else {
                    *grad = -row_val * (*col_val);
                }
            }
        }

        {
            mat_mul(
                1.0,
                gradient,
                &mut jacobian.deref(),
                beta,
                &mut self.operand_gradient.borrow_mut()
            );
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
