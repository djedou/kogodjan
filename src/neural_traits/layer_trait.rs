

pub trait LayerT {

    type Weights;
    type Biases;
    type Outputs;
    /// get the weights of the neurons in the layer
    /// 
    /// # Example
    /// 
    /// ```
    /// use kongodjan::architectures::perceptron::FullConnectedLayer;
    /// use kongodjan::ndarray::{ArrayD};
    /// use kongodjan::neural_traits::LayerT;
    ///
    /// let layer = FullConnectedLayer::<Box<dyn Fn(ArrayD<f64>) -> ArrayD<f64>>>::new(4, 2);
    /// let weights = layer.get_weights();
    ///
    /// assert_eq!(weights.shape(),  &[4,2]);
    /// ```
    /// 
    fn get_weights(&self) -> Self::Weights;

    /// get the biases of the neurons in the layer
    fn get_biases(&self) -> Self::Biases;

    /// get the output of the neurons in the layer
    fn get_outputs(&self) -> Option<Self::Outputs>;
}