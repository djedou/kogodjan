use crate::maths::{Matrix, exp, simd_sum, softmax_exp_sum};
use std::cell::{Ref, RefCell};
use std::ops::{Deref};
use std::rc::Rc;
use crate::node::{PassCounter, Node, ForwardAction, BackwardAction, Bor};
use itertools::izip;


#[derive(Debug)]
pub struct LogSoftmaxNode<OP> {
    value: RefCell<Matrix>,
    operand_gradient: RefCell<Matrix>,
    operand: Rc<OP>,
    needs_gradient: bool,
    counter: PassCounter,
}

impl<OP> LogSoftmaxNode<OP>
where
    OP: Node<Value = Matrix>,
{
    pub fn new(operand: Rc<OP>) -> Self {
        let value = {
            let operand_value = operand.value();
            let operand_slice = operand_value.deref().as_slice().unwrap();
            let max = operand_slice.iter().fold(std::f64::MIN, |x, y| x.max(*y));

            let denominator = max + operand_slice
                .iter()
                .map(|&x| exp(x - max))
                .sum::<f64>()
                .ln();

            operand_value.deref() - denominator
        };

        let gradient = &value * 0.0;
        let needs_gradient = operand.needs_gradient();

        LogSoftmaxNode {
            value: RefCell::new(value),
            operand_gradient: RefCell::new(gradient),
            operand: operand,
            needs_gradient: needs_gradient,
            counter: PassCounter::default(),
        }
    }

    /// An additional method for zeroing the counter for use in the
    /// log-softmax loss, where the actuall log-softmax layer is skipped
    /// when backpropagating.
    pub fn zero_counter(&self) {
        self.counter.clear();
    }
}

impl<OP> Node for LogSoftmaxNode<OP>
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
        dest.assign(self.operand.value().deref());

        let operand_value = self.operand.value();
        let operand_slice = operand_value.deref().as_slice().unwrap();
        let max = operand_slice.iter().fold(std::f64::MIN, |x, y| x.max(*y));

        let denominator = max + softmax_exp_sum(operand_slice, max).ln();

        dest.as_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|x| *x -= denominator);
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        let beta = match self.counter.backward() {
            BackwardAction::Set => 0.0,
            BackwardAction::Increment => 1.0,
        };

        {
            let value = self.value.borrow();
            let value_slice = value.as_slice().expect("Can't get value slice.");

            let gradient_slice = gradient
                .as_slice()
                .expect("Can't get input gradient slice.");
            let mut downstream_gradient = self.operand_gradient.borrow_mut();
            let downstream_gradient_slice = downstream_gradient
                .as_slice_mut()
                .expect("Can't get output gradient slice");

            let gradient_sum = simd_sum(gradient_slice);

            for (out_grad, in_grad, &val) in
                izip!(downstream_gradient_slice, gradient_slice, value_slice)
            {
                *out_grad = beta * (*out_grad) + (*in_grad) - exp(val) * gradient_sum;
            }
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
