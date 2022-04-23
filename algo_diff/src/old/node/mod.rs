
/*
mod add_node;
mod concatinate_node;
mod div_node;
mod dot_node;
mod gradient_accumulator;
mod hogwild_parameter;
mod index_input_node;
mod input_node;
mod mul_node;
mod parameter_node;
mod slice_node;
mod sub_node;
mod operators;
mod vector_dot_node;
mod variable;
mod output_node;


pub use add_node::*;
pub use concatinate_node::*;
pub use div_node::*;
pub use dot_node::*;
pub use gradient_accumulator::*;
pub use hogwild_parameter::*;
pub use index_input_node::*;
pub use input_node::*;
pub use mul_node::*;
pub use parameter_node::*;
pub use slice_node::*;
pub use sub_node::*;
pub use vector_dot_node::*;
pub use operators::*;
pub use variable::*;
pub use output_node::*;


//use add_node::*;
//use concatinate_node::*;

use std::cell::{Cell, Ref};
use std::fmt;
use std::ops::{Deref};
use std::rc::Rc;
use crate::maths::{Matrix, slice_assign, ArraySlice, ArraySliceMut};
use itertools::izip;



#[derive(Debug, PartialEq)]
pub enum ForwardAction {
    Evaluate,
    Cached,
}

#[derive(Debug, PartialEq)]
pub enum BackwardAction {
    Set,
    Increment,
}

#[derive(Debug, Default)]
pub struct PassCounter {
    forward_count: Cell<usize>,
    backward_count: Cell<usize>,
}

impl PassCounter {
    pub fn clear(&self) {
        self.forward_count.set(0);
        self.backward_count.set(0);
    }
    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        debug_assert!(self.recurse_backward(), "Not fully backpropagated.");

        self.forward_count.get() == 0
    }
    pub fn recurse_backward(&self) -> bool {
        let backward_count = self.backward_count.get();
        let forward_count = self.forward_count.get();

        assert!(backward_count <= forward_count);

        if backward_count == forward_count {
            self.clear();
            true
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn forward(&self) -> ForwardAction {
        let count = self.forward_count.get();
        self.forward_count.set(count + 1);

        match count {
            0 => ForwardAction::Evaluate,
            _ => ForwardAction::Cached,
        }
    }
    #[inline(always)]
    pub fn backward(&self) -> BackwardAction {
        let backward_count = self.backward_count.get();

        let action = match backward_count {
            0 => BackwardAction::Set,
            _ => BackwardAction::Increment,
        };

        self.backward_count.set(backward_count + 1);

        action
    }
}

/// Generalisation over borrowed `RefCell` values
/// and simple references.
#[derive(Debug)]
pub enum Bor<'value, T: 'value> {
    /// Ref from a `RefCell`.
    RefGuard(Ref<'value, T>),
    /// Plain reference.
    Reference(&'value T),
}

impl<'value, T: 'value> Deref for Bor<'value, T> {
    type Target = T;
    fn deref(&self) -> &T {
        match *self {
            Bor::RefGuard(ref val) => val.deref(),
            Bor::Reference(ref val) => val.deref(),
        }
    }
}

impl<'value, T: 'value + fmt::Display> fmt::Display for Bor<'value, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.deref())
    }
}

/// Trait representing a computation node. Structs implementing
/// this trait can be used as elements of the computation graph.
pub trait Node: fmt::Debug + 'static {
    /// Type of the node's value.
    type Value;
    /// Type of the input gradient the node receives
    /// during backpropagation.
    type InputGradient;
    /// Perform the forward step. Should recursively call
    /// the forward methods of its ancestors.
    fn forward(&self);
    /// Perform the backward step. Should recursively call
    /// the backward methods of its ancestors.
    fn backward(&self, gradient: &Ref<Self::InputGradient>);
    /// Return the value of the node.
    fn value(&self) -> Bor<Self::Value>;
    /// If the node needs to be used in the backward step.
    fn needs_gradient(&self) -> bool;
    /// Reset the caches of this node and its parents.
    fn clear(&self);
}

impl Node for Rc<dyn Node<Value = Matrix, InputGradient = Matrix>> {
    type Value = Matrix;
    type InputGradient = Matrix;
    fn forward(&self) {
        self.deref().forward()
    }
    fn backward(&self, gradient: &Ref<Self::InputGradient>) {
        self.deref().backward(gradient)
    }
    fn value(&self) -> Bor<Self::Value> {
        self.deref().value()
    }
    fn needs_gradient(&self) -> bool {
        self.deref().needs_gradient()
    }
    fn clear(&self) {
        self.deref().clear()
    }
}
*/


/*
#[cfg(test)]
mod tests {
    use nn;

    use super::*;

    #[test]
    fn test_sub_counter() {
        let x = ParameterNode::new(nn::xavier_normal(1, 1));
        let y = x.clone() - x.clone();

        let mut z = y.clone() + y.clone() + y.clone();

        z.forward();
        assert_eq!(y.node.counter.forward_count.get(), 3);
        z.backward(1.0);
        assert_eq!(y.node.counter.backward_count.get(), 0);
        assert_eq!(y.node.counter.forward_count.get(), 0);
    }
}
*/