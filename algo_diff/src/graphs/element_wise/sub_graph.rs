use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix},

};
use std::fmt::Debug;


/// # Substraction Operator Graph
/// ```
/// use algo_diff::graphs::{SubGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // A - B
/// let mut sub_graph = SubGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
/// 
/// let output = sub_graph.forward([lhs, rhs]);
/// let deriv = sub_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("sub_graph: {:#?}", sub_graph);
/// ```
/// 
#[derive(Debug)]
pub struct SubGraph {
    gradients: [Matrix; 2]
}


impl SubGraph {
    pub fn new() -> SubGraph {

        SubGraph {
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }
}


impl Graph for SubGraph {
    type Input = [Matrix; 2];
    type Gradient = Option<[Matrix; 2]>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let deriv = Matrix::ones((inputs[0].nrows(), inputs[0].ncols()));
        self.gradients = [deriv.clone(), (-1.0 * deriv)];

        inputs[0].clone() - inputs[1].clone()
    }

    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient {
        match gradient {
            Some(grad) => {
                let lhs_grad = self.gradients[0].clone() * grad.clone();
                let rhs_grad = self.gradients[1].clone() * grad;
                Some([lhs_grad, rhs_grad])
            },
            None => {
                Some(self.gradients.clone())
            }
        }
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}






#[cfg(test)]
mod sub_graph_test {
    use super::*;

    #[test]
    fn new() {
        let sub_graph = SubGraph::new();
        dbg!("sub_graph: {:#?}", sub_graph);
    }

    #[test]
    fn forward() {
        let mut sub_graph = SubGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        sub_graph.forward([lhs, rhs]);
        dbg!("sub_graph: {:#?}", sub_graph);
    }

    #[test]
    fn backward() {
        let mut sub_graph = SubGraph::new();

        let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        let output = sub_graph.forward([lhs, rhs]);
        let deriv = sub_graph.backward(None);
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        println!("sub_graph: {:#?}", sub_graph);
    }
}