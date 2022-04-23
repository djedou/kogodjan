use crate::{
    graphs::{
        GraphBuilder,
        Graph
    },
    maths::{Matrix, get_div_deriv},

};
use std::fmt::Debug;


/// # Division Operator Graph
/// ```
/// use algo_diff::graphs::{DivGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // A / B
/// let mut div_graph = DivGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
/// 
/// let output = div_graph.forward([lhs, rhs]);
/// let deriv = div_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("div_graph: {:#?}", div_graph);
/// ```
/// 
#[derive(Debug)]
pub struct DivGraph {
    gradients: [Matrix; 2]
}


impl DivGraph {
    pub fn new() -> DivGraph {

        DivGraph {
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }
}


impl Graph for DivGraph {
    type Input = [Matrix; 2];
    type Gradient = Option<[Matrix; 2]>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        self.gradients = get_div_deriv(&inputs[0], &inputs[1]);

        inputs[0].clone() / inputs[1].clone()
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
mod div_graph_test {
    use super::*;

    #[test]
    fn new() {
        let div_graph = DivGraph::new();
        dbg!("div_graph: {:#?}", div_graph);
    }

    #[test]
    fn forward() {
        let mut div_graph = DivGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        div_graph.forward([lhs, rhs]);
        dbg!("div_graph: {:#?}", div_graph);
    }

    #[test]
    fn backward() {
        let mut div_graph = DivGraph::new();

        let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        let output = div_graph.forward([lhs, rhs]);
        let deriv = div_graph.backward(None);
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        println!("div_graph: {:#?}", div_graph);
    }
}