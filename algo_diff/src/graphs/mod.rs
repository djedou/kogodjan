mod element_wise;
mod operations;

use std::fmt::Debug;
use crate::maths::Matrix;

pub use element_wise::*;
pub use operations::*;




pub trait Graph: Debug {
    type Input;
    type Gradient;
    type Output;

    fn forward(&mut self, input: Self::Input) -> Self::Output;
    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient;
    fn set_builder(&mut self, builder: GraphBuilder);
}


#[derive(Debug)]
pub enum GraphBuilder {
    None,
    AddGraph,
    MulGraph,
    SubGraph,
    DivGraph,
    DotGraph,
    LinearGraph, // function affine AX + B
    SigmoidGraph,
    MseGraph,
    TanhGraph,
    Compose(Box<GraphBuilder>, Box<GraphBuilder>) // (Parent, Child)
}


pub trait StepsBuilder {
    fn build(&mut self);
}