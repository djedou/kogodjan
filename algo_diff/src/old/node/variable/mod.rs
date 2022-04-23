use crate::node::Node;
use crate::maths::Matrix;

use std::cell::RefCell;
use std::clone::Clone;
use std::ops::{Deref, Neg};
use std::rc::Rc;
use crate::node::*;
use crate::clamp;
use crate::{merge_parameters};
use crate::graph::DataInput;


/// Handle to a node in the computation graph. The underlying nodes
/// are reference counted, so the handles can be freely cloned to
/// use the nodes multiple times in the same graph.
#[derive(Debug)]
pub struct Variable<T>
where
    T: Node,
{
    pub node: Rc<T>,
    pub grad: Option<RefCell<Matrix>>,
    pub parameters: Vec<Variable<ParameterNode>>
}


impl<T: Node> Clone for Variable<T> {
    fn clone(&self) -> Self {
        Variable {
            node: Rc::clone(&self.node),
            grad: None,
            parameters: self.parameters.clone(),
        }
    }
}

impl<T> Variable<T>
where
    T: Node,
{
    pub fn new(node: Rc<T>, parameters: Vec<Variable<ParameterNode>>) -> Self {
        Variable {
            node: node,
            grad: None,
            parameters: parameters,
        }
    }
    /// Get the value of the node.
    pub fn value(&self) -> Bor<T::Value> {
        self.node.value()
    }
    /// Run the forward pass through the subgraph terminating at this node,
    /// recursing through the ancestor nodes.
    pub fn forward(&self) {
        self.node.forward()
    }
    /// Clear the graph caches. Must be called whenever inputs change and [backward] is not
    /// called.
    pub fn clear(&self) {
        self.node.clear();
    }

    /// Zero the accumulated gradients for the parameter nodes in this graph.
    pub fn zero_gradient(&self) {
        for param in self.parameters() {
            param.node.zero_gradient();
        }
    }

    /// Return the parameters of the graph.
    pub fn parameters(&self) -> &[Variable<ParameterNode>] {
        &self.parameters[..]
    }

    /// Mutably return the parameters of the graph.
    pub fn parameters_mut(&mut self) -> &mut [Variable<ParameterNode>] {
        &mut self.parameters[..]
    }
}



impl<T> Variable<T>
where
    T: Node<Value = Matrix, InputGradient = Matrix>,
{
    /// Box the variable, erasing its specific type. Use to manage the complexity
    /// of variable types in deep computation graphs.
    pub fn boxed(&self) -> Variable<Rc<dyn Node<Value = Matrix, InputGradient = Matrix>>> {
        Variable::new(
            Rc::new(self.node.clone() as Rc<dyn Node<Value = Matrix, InputGradient = Matrix>>),
            self.parameters.clone(),
        )
    }

    /// Run the backward pass through the subgraph terminating at this node.
    /// The weight parameter scales the gradients.
    pub fn backward(&mut self, weight: f64) {
        let val = self.node.value();

        self.grad
            .get_or_insert_with(|| RefCell::new(val.map(|_| weight)))
            .borrow_mut()
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|x| *x = weight);

        if let Some(ref grad) = self.grad {
            self.node.backward(&grad.borrow());
        }
    }

    /// Clip the value. Useful for clipping losses.
    pub fn clip(&self, min: f64, max: f64) {
        let bor_value = self.node.value();
        let value: &Matrix = bor_value.deref();
        let value = unsafe { &mut *(value as *const Matrix as *mut Matrix) };

        value
            .as_slice_mut()
            .unwrap()
            .iter_mut()
            .for_each(|x| *x = 100.0 * clamp(*x, min, max));
    }

    /// Square this variable.
    pub fn square(&self) -> Variable<SquareNode<T>> {
        Variable::new(
            Rc::new(SquareNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Sum this variable.
    pub fn scalar_sum(&self) -> Variable<SumNode<T>> {
        Variable::new(
            Rc::new(SumNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Take the natural logarithm of this variable.
    pub fn ln(&self) -> Variable<LogNode<T>> {
        Variable::new(
            Rc::new(LogNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Take the tanh of this variable.
    pub fn tanh(&self) -> Variable<TanhNode<T>> {
        Variable::new(
            Rc::new(TanhNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Transpose this variable.
    pub fn t(&self) -> Variable<TransposeNode<T>> {
        Variable::new(
            Rc::new(TransposeNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Exponentiate this variable.
    pub fn exp(&self) -> Variable<ExpNode<T>> {
        Variable::new(
            Rc::new(ExpNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Compute the softmax of this variable.
    pub fn softmax(&self) -> Variable<SoftmaxNode<T>> {
        Variable::new(
            Rc::new(SoftmaxNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Compute the log-softmax of this variable.
    pub fn log_softmax(&self) -> Variable<LogSoftmaxNode<T>> {
        Variable::new(
            Rc::new(LogSoftmaxNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Compute the sigmoid of this variable.
    pub fn sigmoid(&self) -> Variable<SigmoidNode<T>> {
        Variable::new(
            Rc::new(SigmoidNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Compute the ReLU of this variable.
    pub fn relu(&self) -> Variable<ReluNode<T>> {
        Variable::new(
            Rc::new(ReluNode::new(Rc::clone(&self.node))),
            self.parameters.clone(),
        )
    }

    /// Compute the row-wise vector dot product of LHS and RHS.
    pub fn vector_dot<S>(&self, other: &Variable<S>) -> Variable<VectorDotNode<T, S>>
    where
        S: Node<Value = Matrix, InputGradient = Matrix>,
    {
        Variable::new(
            Rc::new(VectorDotNode::new(
                Rc::clone(&self.node),
                Rc::clone(&other.node),
            )),
            merge_parameters(&self.parameters, &other.parameters),
        )
    }

    /// Compute the matrix multiplication of LHS and RHS.
    pub fn dot<S>(&self, other: &Variable<S>) -> Variable<DotNode<T, S>>
    where
        S: Node<Value = Matrix, InputGradient = Matrix>,
    {
        Variable::new(
            Rc::new(DotNode::new(Rc::clone(&self.node), Rc::clone(&other.node))),
            merge_parameters(&self.parameters, &other.parameters),
        )
    }

    /// Stack/concatenate LHS and RHS, either row-wise (`ndarray::Axis(0)`) or
    /// column-wise (`ndarray::Axis(1)`).
    pub fn stack<S>(
        &self,
        other: &Variable<S>,
        axis: ndarray::Axis,
    ) -> Variable<ConcatenateNode<T, S>>
    where
        S: Node<Value = Matrix, InputGradient = Matrix>,
    {
        Variable::new(
            Rc::new(ConcatenateNode::new(
                Rc::clone(&self.node),
                Rc::clone(&other.node),
                axis,
            )),
            merge_parameters(&self.parameters, &other.parameters),
        )
    }

    /// Slice the node according to the `ndarray` slice syntax.
    pub fn slice(
        &self,
        slice: &ndarray::SliceInfo<[ndarray::SliceInfoElem; 2], ndarray::Ix2, ndarray::Ix2>,
    ) -> Variable<SliceNode<T>> {
        Variable::new(
            Rc::new(SliceNode::new(Rc::clone(&self.node), slice)),
            self.parameters.clone(),
        )
    }
}

impl Variable<ParameterNode> {
    /// Return the (dense) gradient value of this node.
    pub fn gradient(&self) -> Matrix {
        self.node.gradient.borrow().materialized_gradient()
    }

    pub fn as_ptr(&self) -> *const ParameterNode {
        self.node.deref() as *const ParameterNode
    }

    /// Row-wise indexing of this parameter node. Primiarily used
    /// to implement embedding layers.
    pub fn index(&self, index: &Variable<IndexInputNode>) -> Variable<IndexNode<ParameterNode>> {
        Variable::new(
            Rc::new(IndexNode::new(
                Rc::clone(&self.node),
                Rc::clone(&index.node),
            )),
            merge_parameters(&self.parameters, &index.parameters),
        )
    }
}

/*
impl<T> Variable<nn::losses::SparseCategoricalCrossentropyNode<T>>
where
    T: Node<Value = Matrix, InputGradient = Matrix>,
{
    /// Return the log-softmax predictions from a sparse categorical
    /// cross-entropy node.
    ///
    /// Calling `.value()` on the node returns the value of the loss;
    /// this function allows getting the predictins with low overhead.
    pub fn predictions(&self) -> Bor<Matrix> {
        self.node.predictions()
    }
}
*/
impl<'value> DataInput<&'value Matrix> for Variable<ParameterNode> {
    fn set_value(&self, value: &Matrix) {
        let param_value = unsafe { &mut *(self.node.value.deref().value.as_ptr()) };
        param_value.assign(value)
    }
}

impl<'value> DataInput<&'value Matrix> for Variable<InputNode> {
    fn set_value(&self, value: &Matrix) {
        self.node.value.borrow_mut().assign(value);
    }
}

impl DataInput<f64> for Variable<InputNode> {
    fn set_value(&self, value: f64) {
        self.node.value.borrow_mut()[(0, 0)] = value;
    }
}

impl<'value> DataInput<&'value [usize]> for Variable<IndexInputNode> {
    fn set_value(&self, value: &[usize]) {
        let mut node_value = self.node.value.borrow_mut();
        node_value.clear();
        node_value.extend_from_slice(value);
    }
}

impl DataInput<usize> for Variable<IndexInputNode> {
    fn set_value(&self, value: usize) {
        let mut node_value = self.node.value.borrow_mut();
        node_value.clear();
        node_value.push(value);
    }
}



impl<T> Neg for Variable<T>
where
    T: Node<Value = Matrix, InputGradient = Matrix>,
{
    type Output = Variable<NegNode<T>>;
    fn neg(self) -> Self::Output {
        Variable::new(Rc::new(NegNode::new(self.node)), self.parameters.clone())
    }
}
