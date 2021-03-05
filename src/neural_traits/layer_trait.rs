
use crate::maths::types::MatrixD;
use crate::optimizers::Optimizer;
use crate::utils::Parameters;
use std::fmt::Debug;
//use crate::errors::{Gradient};

pub trait LayerT: Debug {
    /*/// get the weights of the neurons in the layer
    /// 
    /// # Example
    /// 
    /// ```
    /// use kongodjan::architectures::perceptron::FullConnectedLayer;
    /// use kongodjan::ndarray::{ArrayD};
    /// use kongodjan::neural_traits::LayerT;
    /// use kongodjan::activations::{tansig, tansig_deriv};
    ///
    /// let layer = FullConnectedLayer::new(4, 2, tansig, tansig_deriv);
    /// let weights = layer.get_weights();
    ///
    /// assert_eq!(weights.shape(),  &[4,2]);
    /// ```
    /// 
    fn get_weights(&self) -> Array2<f64>;

    /// get the biases of the neurons in the layer
    fn get_biases(&self) -> Array2<f64>;*/

    fn forward(&mut self, inputs: &MatrixD<f64>) -> MatrixD<f64>;

    fn backward(&mut self, lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, optimizer: &Optimizer) -> MatrixD<f64>;

    fn get_layer_id(&self) -> i32 {0}

    fn set_weights(&mut self, weights: MatrixD<f64>);

    fn set_biases(&mut self, biases: MatrixD<f64>);

    fn save(&self) -> Option<Parameters> {
        None
    }

    fn get_weights(&self) -> Option<MatrixD<f64>>;

    fn get_biases(&self) -> Option<MatrixD<f64>>;

}