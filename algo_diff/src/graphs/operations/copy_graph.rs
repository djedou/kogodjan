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
/// use algo_diff::graphs::{CopyGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // AB
/// // Where: A is (m,n) Matrix and B is (n,p) Matrix.
/// let mut copy_graph = CopyGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
/// 
/// let output = copy_graph.forward([lhs, rhs]);
/// let deriv = copy_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("copy_graph: {:#?}", copy_graph);
/// ```
/// 
#[derive(Debug)]
pub struct CopyGraph {
    output_size: usize
}


impl CopyGraph {
    pub fn new(size: usize) -> CopyGraph {

        CopyGraph {
            output_size: size
        }
    }
}


impl Graph for CopyGraph {
    type Input = Matrix;
    type Gradient = Option<Matrix>;
    type Output = Vec<Matrix>;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let mut outputs: Vec<Matrix> = vec![];
        for _ in 0..self.output_size {
            outputs.push(inputs.clone());
        }

        outputs
    }

    fn backward(&mut self, _gradient: Option<Matrix>) -> Self::Gradient {
        None
    }

    fn backward_with_more_gradients(&mut self, gradients: Option<&[Matrix]>) -> Self::Gradient {
        match gradients {
            Some(grad) => {
                let mut gradient = grad[0].clone();

                for i in 1..grad.len() {
                    gradient = gradient + grad[i].clone();
                }

                Some(gradient)
            },
            None => {
                None
            }
        }
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}





#[cfg(test)]
mod copy_graph_test {
    use super::*;

    #[test]
    fn new() {
        let copy_graph = CopyGraph::new(4);
        dbg!("copy_graph: {:#?}", copy_graph);
    }

    #[test]
    fn forward() {
        let mut copy_graph = CopyGraph::new(2);

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        
        copy_graph.forward(lhs);
        dbg!("copy_graph: {:#?}", copy_graph);
    }

    #[test]
    fn backward() {
        let mut copy_graph = CopyGraph::new(2);

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        let grad1 = Matrix::from_shape_vec((4,1), vec![1., 2., 1., 1.]).unwrap();
        let grad2 = Matrix::from_shape_vec((4,1), vec![1., 2., 0., 1.]).unwrap();
        
        let output = copy_graph.forward(lhs);
        let deriv = copy_graph.backward_with_more_gradients(Some(&[grad1, grad2]));
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        //println!("copy_graph: {:#?}", copy_graph);
    }
}