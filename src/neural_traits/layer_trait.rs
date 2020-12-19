
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
    fn get_weights(&self) -> Array2<f32>;

    /// get the biases of the neurons in the layer
    fn get_biases(&self) -> Array2<f32>;*/

    fn forward(&mut self, inputs: &MatrixD<f32>) -> MatrixD<f32>;

    fn backward(&mut self, lr: &f32, batch_size: &usize, gradient: &MatrixD<f32>, optimizer: &Optimizer) -> MatrixD<f32>;

    fn get_layer_id(&self) -> i32 {0}

    fn set_weights(&mut self, weights: MatrixD<f32>);

    fn set_biases(&mut self, biases: MatrixD<f32>);

    fn save(&self) -> Option<Parameters> {
        None
    }

    fn get_weights(&self) -> Option<MatrixD<f32>>;

    fn get_biases(&self) -> Option<MatrixD<f32>>;

}