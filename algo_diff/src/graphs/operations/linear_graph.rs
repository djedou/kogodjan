use crate::{
    graphs::{
        GraphBuilder, StepsBuilder,
        Graph, DotGraph, AddGraph,
    },
    maths::{Matrix},

};
use std::collections::HashMap;
use std::fmt::Debug;


/// # Linear Operator Graph
/// ```
/// use algo_diff::graphs::{LinearGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // WX + B 
/// // Where: W is (m,n) Matrix, X is (m,p) Matrix and B is (n,p) Matrix.
/// let mut linear_graph = LinearGraph::new();
///
/// let w = Matrix::from_shape_vec((2,2), vec![1., 0., 0., -1.]).unwrap();
/// let w_t = w.reversed_axes();
/// let x = Matrix::from_shape_vec((2,1), vec![1., 2.]).unwrap();
/// let b = Matrix::from_shape_vec((2,1), vec![1., 1.]).unwrap();
/// 
/// let output = linear_graph.forward([w_t, x, b]);
/// let deriv = linear_graph.backward(None);
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv.unwrap());
/// println!("linear_graph: {:#?}", linear_graph);
/// ```
/// 
#[derive(Debug)]
pub struct LinearGraph {
    steps: HashMap<usize, Box<dyn Graph<Input = [Matrix; 2], Gradient = Option<[Matrix; 2]>, Output = Matrix>>>,
    builder: GraphBuilder,
    gradients: [Matrix; 3]
}


impl LinearGraph {
    pub fn new() -> LinearGraph {

        let mut linear_graph = LinearGraph {
            steps: HashMap::new(),
            builder: GraphBuilder::LinearGraph,
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        };
        linear_graph.build();
        linear_graph
    }
}


impl Graph for LinearGraph {
    type Input = [Matrix; 3];
    type Gradient = Option<[Matrix; 3]>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        // Step 1
        let op_1 = self.steps.get_mut(&1).unwrap();
        let op_1_output = op_1.forward([inputs[0].clone(), inputs[1].clone()]);

        // Step 2
        let op_2 = self.steps.get_mut(&2).unwrap();
        let op_2_output = op_2.forward([op_1_output, inputs[2].clone()]);

        op_2_output
    }

    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient {
        match gradient {
            Some(grad) => {
                // Step 2
                let op_2 = self.steps.get_mut(&2).unwrap();
                let op_2_grad = op_2.backward(Some(grad)).unwrap();
                self.gradients.as_mut()[2] = op_2_grad[1].clone();
                
                // Step 1
                let op_1 = self.steps.get_mut(&1).unwrap();
                let op_1_grad = op_1.backward(Some(op_2_grad[0].clone())).unwrap();
                self.gradients.as_mut()[0] = op_1_grad[0].clone();
                self.gradients.as_mut()[1] = op_1_grad[1].clone();
                Some(self.gradients.clone())
            },
            None => {
                // Step 2
                let op_2 = self.steps.get_mut(&2).unwrap();
                let op_2_grad = op_2.backward(None).unwrap();
                self.gradients.as_mut()[2] = op_2_grad[1].clone();
                
                // Step 1
                let op_1 = self.steps.get_mut(&1).unwrap();
                let op_1_grad = op_1.backward(Some(op_2_grad[0].clone())).unwrap();
                self.gradients.as_mut()[0] = op_1_grad[0].clone();
                self.gradients.as_mut()[1] = op_1_grad[1].clone();
                Some(self.gradients.clone())
            }
        }
    }

    fn backward_with_more_gradients(&mut self, _gradients: Option<&[Matrix]>) -> Self::Gradient {
        None
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}


impl StepsBuilder for LinearGraph {
    fn build(&mut self) {
        match &self.builder {
            GraphBuilder::LinearGraph => {
                self.steps.insert(1, Box::new(DotGraph::new()));
                self.steps.insert(2, Box::new(AddGraph::new()));
            },
            _ => {}
        }
    }
}






#[cfg(test)]
mod linear_graph_test {
    use super::*;

    #[test]
    fn new() {
        let linear_graph = LinearGraph::new();
        dbg!("linear_graph: {:#?}", linear_graph);
    }

    #[test]
    fn forward() {
        let mut linear_graph = LinearGraph::new();

        let a = Matrix::from_shape_vec((2,2), vec![1., 0., 0., -1.]).unwrap();
        let a_t = a.reversed_axes();
        let b = Matrix::from_shape_vec((2,1), vec![1., 2.]).unwrap();
        let c = Matrix::from_shape_vec((2,1), vec![1., 1.]).unwrap();
        
        let output = linear_graph.forward([a_t, b, c]);
        println!("output: {:#?}", output);
        //println!(" ");
        //println!("deriv: {:#?}", deriv);
        //println!("linear_graph: {:#?}", linear_graph);
    }

    #[test]
    fn backward() {
        let mut linear_graph = LinearGraph::new();

        let a = Matrix::from_shape_vec((2,2), vec![1., 0., 0., -1.]).unwrap();
        let a_t = a.reversed_axes();
        let b = Matrix::from_shape_vec((2,1), vec![1., 2.]).unwrap();
        let c = Matrix::from_shape_vec((2,1), vec![1., 1.]).unwrap();
        
        let output = linear_graph.forward([a_t, b, c]);
        let deriv = linear_graph.backward(None);
        println!("output: {:#?}", output);
        //println!(" ");
        println!("deriv: {:#?}", deriv.unwrap());
        //println!("linear_graph: {:#?}", linear_graph);
    }
}