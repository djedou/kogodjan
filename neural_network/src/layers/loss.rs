use algo_diff::{
    maths::{Matrix},
    graphs::{Graph, MseGraph}
};
use crate::loss::Loss;


#[derive(Debug)]
pub struct LossLayer {
    loss: Box<dyn Graph<Input = [Matrix; 2], Gradient = Option<Matrix>, Output = f64>>
}



impl LossLayer {
    
    pub fn new(loss: Loss) -> LossLayer {
        let error: Box<dyn Graph<Input = [Matrix; 2], Gradient = Option<Matrix>, Output = f64>> = match loss {
            Loss::Mse => Box::new(MseGraph::new())
        };

        LossLayer {
            loss: error
        }
    }

    pub fn forward(&mut self, target: Matrix, output: Matrix) -> f64 {
        self.loss.forward([target, output])
    }

    pub fn backward(&mut self) -> Option<Matrix> {
        self.loss.backward(None)
    }
}
