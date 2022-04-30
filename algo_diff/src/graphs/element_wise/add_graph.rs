use crate::{
    graphs::{
        GraphBuilder,
        Graph,
    },
    maths::{Matrix},

};
use std::fmt::Debug;


/// # Addition Operator Graph
/// ```
/// use algo_diff::graphs::{AddGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // A + B
/// let mut add_graph = AddGraph::new();
/// 
/// let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
/// let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
/// 
/// let output = add_graph.forward([lhs, rhs]);
/// let deriv = add_graph.backward(None);
/// 
/// println!("output: {:#?}", output);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// println!(" ");
/// println!("add_graph: {:#?}", add_graph);
/// ```
/// 
#[derive(Debug,)]
pub struct AddGraph {
    gradients: [Matrix; 2]
}


impl AddGraph {
    pub fn new() -> AddGraph {

        AddGraph {
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }
}


impl Graph for AddGraph {
    type Input = [Matrix; 2];
    type Gradient = Option<[Matrix; 2]>;
    type Output = Matrix;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let deriv = Matrix::ones((inputs[0].nrows(), inputs[0].ncols()));
        self.gradients = [deriv.clone(), deriv];

        inputs[0].clone() + inputs[1].clone()
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
mod add_graph_test {
    use super::*;

    #[test]
    fn new() {
        let add_graph = AddGraph::new();
        dbg!("add_graph: {:#?}", add_graph);
    }

    #[test]
    fn forward() {
        let mut add_graph = AddGraph::new();

        let lhs = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        add_graph.forward([lhs, rhs]);
        dbg!("add_graph: {:#?}", add_graph);
    }

    #[test]
    fn backward() {
        let mut add_graph = AddGraph::new();

        let lhs = Matrix::from_shape_vec((2,4), vec![1., 0., 0., 4., 0., 1., 0., 1.]).unwrap();
        let rhs = Matrix::from_shape_vec((2,4), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        let output = add_graph.forward([lhs, rhs]);
        let deriv = add_graph.backward(None);
        println!("output: {:#?}", output);
        println!(" ");
        println!("deriv: {:#?}", deriv);

        println!("add_graph: {:#?}", add_graph);
    }
}