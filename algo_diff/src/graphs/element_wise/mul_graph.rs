use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix},

};
use std::fmt::Debug;


/// # Multiplication Operator Graph
/// ```
/// use algo_diff::graphs::{MulGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // A * B
/// let mut mul_graph = MulGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
/// 
/// let output = mul_graph.forward([lhs, rhs]);
/// let deriv = mul_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("mul_graph: {:#?}", mul_graph);
/// ```
/// 
#[derive(Debug)]
pub struct MulGraph {
    gradients: [Matrix; 2]
}


impl MulGraph {
    pub fn new() -> MulGraph {

        MulGraph {
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }
}


impl Graph for MulGraph {
    type Input = [Matrix; 2];
    type Gradient = Option<[Matrix; 2]>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        self.gradients = [inputs[1].clone(), inputs[0].clone()];
        inputs[0].clone() * inputs[1].clone()
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

    fn backward_with_more_gradients(&mut self, _gradients: Option<&[Matrix]>) -> Self::Gradient {
        None
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}







#[cfg(test)]
mod mul_graph_test {
    use super::*;

    #[test]
    fn new() {
        let mul_graph = MulGraph::new();
        dbg!("mul_graph: {:#?}", mul_graph);
    }

    #[test]
    fn forward() {
        let mut mul_graph = MulGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        mul_graph.forward([lhs, rhs]);
        dbg!("mul_graph: {:#?}", mul_graph);
    }

    #[test]
    fn backward() {
        let mut mul_graph = MulGraph::new();

        let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        let output = mul_graph.forward([lhs, rhs]);
        let deriv = mul_graph.backward(None);
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        println!("mul_graph: {:#?}", mul_graph);
    }
}