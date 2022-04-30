use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix, sigmoid},

};
use std::fmt::Debug;


/// # Sigmoid Operator Graph
/// ```
/// use algo_diff::graphs::{SigmoidGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // 1.0 / 1.0 + e^(-x)
/// let mut sigmoid_graph = SigmoidGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// 
/// let output = sigmoid_graph.forward(lhs);
/// let deriv = sigmoid_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("sigmoid_graph: {:#?}", sigmoid_graph);
/// ```
/// 
#[derive(Debug)]
pub struct SigmoidGraph {
    gradients: Matrix
}


impl SigmoidGraph {
    pub fn new() -> SigmoidGraph {

        SigmoidGraph {
            gradients: Matrix::zeros((1,1))
        }
    }
}


impl Graph for SigmoidGraph {
    type Input = Matrix;
    type Gradient = Option<Matrix>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let mut value = inputs.clone();
        value.mapv_inplace(|x| sigmoid(x));
        
        let mut sig = 1.0 - inputs;
        sig.mapv_inplace(|x| sigmoid(x));
        self.gradients = value.clone() * sig;

        value
    }

    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient {
        match gradient {
            Some(grad) => {
                let new_grad = self.gradients.clone() * grad;
                Some(new_grad)
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
mod sigmoid_graph_test {
    use super::*;

    #[test]
    fn new() {
        let sigmoid_graph = SigmoidGraph::new();
        dbg!("sigmoid_graph: {:#?}", sigmoid_graph);
    }

    #[test]
    fn forward() {
        let mut sigmoid_graph = SigmoidGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        
        sigmoid_graph.forward(lhs);
        dbg!("sigmoid_graph: {:#?}", sigmoid_graph);
    }

    #[test]
    fn backward() {
        let mut sigmoid_graph = SigmoidGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        
        let output = sigmoid_graph.forward(lhs);
        let deriv = sigmoid_graph.backward(None);
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        println!("sigmoid_graph: {:#?}", sigmoid_graph);
    }
}