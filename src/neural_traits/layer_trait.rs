
//use crate::maths::types::MatrixD;
use crate::maths::Matrix;
use crate::activators::types::{ActivatorDeriv};
//use crate::utils::Parameters;
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

    fn forward(&mut self, inputs: &Matrix<f64>) -> Result<Matrix<f64>, String>;

    fn backward(&mut self, lr: f64, gradient: &Matrix<f64>);

    fn update_parameters(&mut self);

    fn get_layer_id(&self) -> i32 {0}

    fn set_weights(&mut self, weights: Matrix<f64>);

    fn set_biases(&mut self, biases: Matrix<f64>);

    /*fn save(&self) -> Option<Parameters> {
        None
    }*/

    fn get_weights(&self) -> Option<Matrix<f64>>;

    fn get_inputs(&self) -> Option<Matrix<f64>>;

    fn get_biases(&self) -> Option<Matrix<f64>>;

    fn get_net_inputs(&self) -> Option<Matrix<f64>>;

    fn get_activator_deriv(&self) -> Option<ActivatorDeriv>;

}
