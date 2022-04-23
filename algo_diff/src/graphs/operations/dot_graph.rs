use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix},
};
use std::fmt::Debug;


/// # Dot Operator Graph
/// ```
/// use algo_diff::graphs::{DotGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // AB
/// // Where: A is (m,n) Matrix and B is (n,p) Matrix.
/// let mut dot_graph = DotGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
/// 
/// let output = dot_graph.forward([lhs, rhs]);
/// let deriv = dot_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("dot_graph: {:#?}", dot_graph);
/// ```
/// 
#[derive(Debug)]
pub struct DotGraph {
    inputs: [Matrix; 2]
}


impl DotGraph {
    pub fn new() -> DotGraph {

        DotGraph {
            inputs: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }
}


impl Graph for DotGraph {
    type Input = [Matrix; 2];
    type Gradient = Option<[Matrix; 2]>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        self.inputs = inputs.clone();
        inputs[0].dot(&inputs[1])
    }

    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient {
        match gradient {
            Some(grad) => {
                let rhs_deriv = self.inputs[0].clone().reversed_axes().dot(&grad);
                let lhs_deriv = grad.dot(&self.inputs[1].clone().reversed_axes());
                
                Some([lhs_deriv, rhs_deriv])
            },
            None => {
                let lhs_deriv = self.inputs[1].clone().reversed_axes();
                let rhs_deriv = self.inputs[0].clone().reversed_axes();
                
                Some([lhs_deriv, rhs_deriv])
            }
        }
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}





#[cfg(test)]
mod dot_graph_test {
    use super::*;

    #[test]
    fn new() {
        let dot_graph = DotGraph::new();
        dbg!("dot_graph: {:#?}", dot_graph);
    }

    #[test]
    fn forward() {
        let mut dot_graph = DotGraph::new();

        let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        dot_graph.forward([lhs, rhs]);
        dbg!("dot_graph: {:#?}", dot_graph);
    }

    #[test]
    fn backward() {
        let mut dot_graph = DotGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((2,4), vec![1., 2., 1., 1., 1., 2., 0., 1.]).unwrap();
        
        let _output = dot_graph.forward([lhs, rhs]);
        let _deriv = dot_graph.backward(None);
        //println!("output: {:#?}", output);
        //println!(" ");
        //println!("deriv: {:#?}", deriv);

        println!("dot_graph: {:#?}", dot_graph);
    }
}