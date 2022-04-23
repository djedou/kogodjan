use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix},

};
use std::fmt::Debug;


/// # Sigmoid Operator Graph
/// ```
/// use algo_diff::graphs::{TanhGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // 1.0 / 1.0 + e^(-x)
/// let mut tanh_graph = TanhGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// 
/// let output = tanh_graph.forward(lhs);
/// let deriv = tanh_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("tanh_graph: {:#?}", tanh_graph);
/// ```
/// 
#[derive(Debug)]
pub struct TanhGraph {
    gradients: Matrix
}


impl TanhGraph {
    pub fn new() -> TanhGraph {

        TanhGraph {
            gradients: Matrix::zeros((1,1))
        }
    }
}


impl Graph for TanhGraph {
    type Input = Matrix;
    type Gradient = Option<Matrix>;
    type Output = Matrix;



    fn forward(&mut self, _inputs: Self::Input) -> Self::Output {
        /*
        let op = self.steps.get_mut(&1).unwrap();
        op.forward(&[inputs.clone()]);
        let value = op.get_value();
        self.gradients = get_sigmoid_deriv(&value.clone());
        value
        */
        Matrix::zeros((1,1))
    }

    fn backward(&mut self, _gradient: Option<Matrix>) -> Self::Gradient {
        /*match gradient {
            Some(grad) => {
                let new_grad = self.gradients.clone() * grad;
                Some(new_grad)
            },
            None => {
                Some(self.gradients.clone())
            }
        }*/
        Some(self.gradients.clone())
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}






#[cfg(test)]
mod tanh_graph_test {
    use super::*;

    #[test]
    fn new() {
        let tanh_graph = TanhGraph::new();
        dbg!("tanh_graph: {:#?}", tanh_graph);
    }

    #[test]
    fn forward() {
        let mut tanh_graph = TanhGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        
        tanh_graph.forward(lhs);
        dbg!("tanh_graph: {:#?}", tanh_graph);
    }

    #[test]
    fn backward() {
        let mut tanh_graph = TanhGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        
        let output = tanh_graph.forward(lhs);
        let deriv = tanh_graph.backward(None);
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        //println!("tanh_graph: {:#?}", tanh_graph);
    }
}