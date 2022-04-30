use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix, tanh, pow2},

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



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let mut value = inputs.clone();
        value.mapv_inplace(|x| tanh(x));
        
        let mut value_pow2 = value.clone();
        value_pow2.mapv_inplace(|x| pow2(x));
        let deriv = 1.0 - value_pow2;
        
        self.gradients = deriv;

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