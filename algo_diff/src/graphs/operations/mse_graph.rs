use crate::{
    graphs::{
        GraphBuilder, StepsBuilder,
        Graph, SubGraph,
    },
    maths::{Matrix},

};
use std::collections::HashMap;
use std::fmt::Debug;


/// # Mse Operator Graph
/// ```
/// use algo_diff::graphs::{MseGraph, Graph};
/// use algo_diff::maths::Matrix;
/// 
/// // 1.0/N * Sum((target - output)^2)
/// let mut mse_graph = MseGraph::new();
/// 
/// let target = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
/// let output = Matrix::from_shape_vec((4,2), vec![1., 2., 1., 1., 1., 2., 0., 1.]).unwrap();
/// 
/// let loss = mse_graph.forward([target, output]);
/// let deriv = mse_graph.backward(None);
/// println!("loss: {:#?}", loss);
/// println!(" ");
/// println!("deriv: {:#?}", deriv);
/// 
/// println!("mse_graph: {:#?}", mse_graph);
/// ```
/// 
#[derive(Debug)]
pub struct MseGraph {
    steps: HashMap<usize, Box<dyn Graph<Input = [Matrix; 2], Gradient = Option<[Matrix; 2]>, Output = Matrix>>>,
    builder: GraphBuilder,
    gradients: Matrix
}   


impl MseGraph {
    pub fn new() -> MseGraph {

        let mut mse_graph = MseGraph {
            steps: HashMap::new(),
            builder: GraphBuilder::MseGraph,
            gradients: Matrix::zeros((1,1))
        };
        mse_graph.build();
        mse_graph
    }
}


impl Graph for MseGraph {
    type Input = [Matrix; 2];
    type Gradient = Option<Matrix>;
    type Output = f64;



    fn forward(&mut self, inputs: Self::Input) -> Self::Output {
        let mut res = Vec::new();
    
        let loss = |targ: &[f64], out: &[f64]| -> f64 {
            let mut sum = 0.0;
            for i in 0..targ.len() {
                sum += (targ[i] - out[i]).exp2();
            }
            sum / targ.len() as f64
        };
        
        for i in 0..inputs[0].ncols() {
            res.push(loss(&inputs[0].column(i).to_vec(), &inputs[1].column(i).to_vec()));
        }
        let size = res.len() as f64;
        let sum: f64 = res.into_iter().sum();
    
        self.gradients = inputs[0].clone() - inputs[1].clone();

        sum / size
    }

    fn backward(&mut self, gradient: Option<Matrix>) -> Self::Gradient {
        match gradient {
            Some(_grad) => {
                None
            },
            None => {
                Some(-2.0 * self.gradients.clone())
            }
        }
    }

    fn set_builder(&mut self, _builder: GraphBuilder) {}
}


impl StepsBuilder for MseGraph {
    fn build(&mut self) {
        match &self.builder {
            GraphBuilder::MseGraph => {
                self.steps.insert(1, Box::new(SubGraph::new()));
            },
            _ => {}
        }
    }
}






#[cfg(test)]
mod mse_graph_test {
    use super::*;

    #[test]
    fn new() {
        let mse_graph = MseGraph::new();
        dbg!("mse_graph: {:#?}", mse_graph);
    }

    #[test]
    fn forward() {
        let mut mse_graph = MseGraph::new();

        let target = Matrix::from_shape_vec((4,2), vec![1., 0., 0., 1., 0., 1., 0., 1.]).unwrap();
        let output = Matrix::from_shape_vec((4,2), vec![1., 0., 1., 1., 1., 1., 0., 1.]).unwrap();
        
        mse_graph.forward([target, output]);
        dbg!("mse_graph: {:#?}", mse_graph);
    }

    #[test]
    fn backward() {
        let mut mse_graph = MseGraph::new();

        let target = Matrix::from_shape_vec((4,2), vec![1., 4., 1., 3., 0., 1., 0., 1.]).unwrap();
        let output = Matrix::from_shape_vec((4,2), vec![1., 2., 1., 1., 1., 2., 0., 1.]).unwrap();
        
        let loss = mse_graph.forward([target, output]);
        let deriv = mse_graph.backward(None);
        println!("loss: {:#?}", loss);
        //println!(" ");
        println!("deriv: {:#?}", deriv);

        println!("mse_graph: {:#?}", mse_graph);
    }
}


/*
pub fn mse_loss(output: &[Array2<f64>], target: &[Array2<f64>]) -> Vec<f64> {
    let mut res = Vec::new();

    let loss = |out: &[f64], targ: &[f64]| -> f64 {
        let mut sum = 0.0;
        for i in 0..targ.len() {
            sum += (targ[i] - out[i]).exp2();
        }
        sum / targ.len() as f64
    };
    
    for i in 0..target.len() {
        res.push(loss(&output[i].column(0).to_vec(), &target[i].column(0).to_vec()));
    }
    
    res
}







*/