
use ndarray::ArrayD;

pub trait LayerT {
    /// get the weights of the neurons in the layer
    /// 
    /// # Example
    /// 
    /// ```
    /// use kongodjan::architectures::perceptron::FullConnectedLayer;
    /// use kongodjan::ndarray::{ArrayD};
    /// use kongodjan::neural_traits::LayerT;
    /// use kongodjan::activations::tansig;
    ///
    /// let layer = FullConnectedLayer::new(4, 2, tansig);
    /// let weights = layer.get_weights();
    ///
    /// assert_eq!(weights.shape(),  &[4,2]);
    /// ```
    /// 
    fn get_weights(&self) -> ArrayD<f64>;

    /// get the biases of the neurons in the layer
    fn get_biases(&self) -> ArrayD<f64>;

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64>;

    fn backward(&mut self, inputs: ArrayD<f64>);

}