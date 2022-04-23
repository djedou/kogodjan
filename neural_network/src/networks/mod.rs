mod feed_forward;


pub use feed_forward::*;
use algo_diff::maths::Matrix;


pub trait NetworkT {
    //fn run_train(&mut self, features: &[Matrix]) -> Vec<Variable<Rc<dyn Node<Value = Matrix, InputGradient = Matrix>>>>;
    //fn backward_propagation(&mut self, gradient: &mut [Variable<Rc<dyn Node<Value = Matrix, InputGradient = Matrix>>>], lr: f64);
    //fn update_parameters(&mut self, lr: f64);
    /*
    fn predict_forward_propagation(&mut self, features: Vec<Array2<f64>>);
    */
    fn train(&mut self, network_inputs: &[Matrix], network_outputs: &[Matrix], lr: f64, momentum: f64, batch_size: usize, epoch: i32);
    fn predict(&mut self, input: &Matrix) -> Matrix;
    //fn network_outpout(&self) -> Vec<Array2<f64>>;
}
