use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix, softmax, Array},
};
use std::fmt::Debug;


/// # Softmax Operator Graph (Please use it for cross_entropy_graph)
/// ```
/// use algo_diff::graphs::{SoftmaxGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// let mut softmax_graph = SoftmaxGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((8,1), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// 
/// let output = softmax_graph.forward(lhs);
/// let deriv = softmax_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("softmax_graph: {:#?}", softmax_graph);
/// ```
/// 
#[derive(Debug)]
pub struct SoftmaxGraph;


impl SoftmaxGraph {
    pub fn new() -> SoftmaxGraph {

        SoftmaxGraph
    }
}


impl Graph for SoftmaxGraph {
    type Input = Matrix;
    type Gradient = Option<Matrix>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let mut outputs = Matrix::zeros((inputs.nrows(), 0));
        for c in 0..inputs.ncols() {
            let res = softmax(&inputs.column(c).to_vec());
            outputs.push_column(Array::from(res).view()).unwrap();
        }

        outputs
    }

    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient {
        gradient
    }

    fn backward_with_more_gradients(&mut self, _gradients: Option<&[Matrix]>) -> Self::Gradient {
        None
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}





#[cfg(test)]
mod softmax_graph_test {
    use super::*;

    #[test]
    fn new() {
        let softmax_graph = SoftmaxGraph::new();
        dbg!("softmax_graph: {:#?}", softmax_graph);
    }

    #[test]
    fn forward() {
        let mut softmax_graph = SoftmaxGraph::new();

        let lhs = Matrix::from_shape_vec((8,1), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        
        let res = softmax_graph.forward(lhs);
        println!("softmax_graph: {:#?}", res);
    }

    #[test]
    fn backward() {
        /*let mut softmax_graph = SoftmaxGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((2,4), vec![1., 2., 1., 1., 1., 2., 0., 1.]).unwrap();
        
        let _output = softmax_graph.forward([lhs, rhs]);
        let _deriv = softmax_graph.backward(None);
        //println!("output: {:#?}", output);
        //println!(" ");
        //println!("deriv: {:#?}", deriv);

        println!("softmax_graph: {:#?}", softmax_graph);*/
    }
}
