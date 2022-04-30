use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix, ln},
};
use std::fmt::Debug;


/// # Cross Entropy Operator Graph
/// ```
/// use algo_diff::graphs::{CrossEntropyGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // AB
/// // Where: A is (m,n) Matrix and B is (n,p) Matrix.
/// let mut cross_entropy_graph = CrossEntropyGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
/// 
/// let output = cross_entropy_graph.forward([lhs, rhs]);
/// let deriv = cross_entropy_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("cross_entropy_graph: {:#?}", cross_entropy_graph);
/// ```
/// 
#[derive(Debug)]
pub struct CrossEntropyGraph {
    gradients: Matrix
}


impl CrossEntropyGraph {
    pub fn new() -> CrossEntropyGraph {

        CrossEntropyGraph {
            gradients: Matrix::zeros((1,1))
        }
    }
}


impl Graph for CrossEntropyGraph {
    type Input = [Matrix; 2];
    type Gradient = Option<Matrix>;
    type Output = f64;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let mut res: Vec<f64> = vec![];
        
        self.gradients = inputs[1].clone() - inputs[0].clone();
        
        for c in 0..inputs[0].ncols() {
            let target = inputs[0].column(c).to_vec();
            let output = inputs[1].column(c).to_vec();

            let targ_out: Vec<f64> = target.iter().zip(output.iter()).map(|(t,o)| t * ln(*o)).collect();
            res.push(targ_out.iter().sum());
        }
        
        let sum: f64 = res.iter().sum();
        
        -sum
    }

    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient {
        match gradient {
            Some(grad) => {
                Some(self.gradients.clone() * grad)
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
mod cross_entropy_graph_test {
    use super::*;

    #[test]
    fn new() {
        let cross_entropy_graph = CrossEntropyGraph::new();
        dbg!("cross_entropy_graph: {:#?}", cross_entropy_graph);
    }

    #[test]
    fn forward() {
        let mut cross_entropy_graph = CrossEntropyGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        cross_entropy_graph.forward([lhs, rhs]);
        println!("cross_entropy_graph: {:#?}", cross_entropy_graph);
    }

    #[test]
    fn backward() {
        let mut cross_entropy_graph = CrossEntropyGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 2., 1., 1., 1., 2., 0., 1.]).unwrap();
        
        let output = cross_entropy_graph.forward([lhs, rhs]);
        let deriv = cross_entropy_graph.backward(None);
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        //println!("cross_entropy_graph: {:#?}", cross_entropy_graph);
    }
}