
use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Debug, Clone)]
pub struct FullConnectedLayer {
    weights: ArrayD<f64>,
    biases: ArrayD<f64>,
    outputs: Option<ArrayD<f64>>,
}

impl FullConnectedLayer {
    /// create a new full connected layer  
    /// n_inputs is the number of inputs for each neurons  
    /// n_neurons is the number of neurons in the layer  
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        let weights = ArrayD::random(IxDyn(&[n_inputs, n_neurons]), Uniform::new(-1., 1.));
        let biases = ArrayD::random(IxDyn(&[1, n_neurons]), Uniform::new(-1., 1.));
        FullConnectedLayer {
            weights,
            biases,
            outputs: None,
        }
    }

    /// get the weights of the neurons in the layer
    /// 
    /// # Example
    /// 
    /// ```
    /// use kongodjan::architectures::perceptron::FullConnectedLayer;
    /// 
    /// let layer = FullConnectedLayer::new(4, 2);
    /// let weights = layer.get_weights();
    ///
    /// assert_eq!(weights.shape(),  &[4,2]);
    /// ```
    /// 
    pub fn get_weights(&self) -> ArrayD<f64> {
        self.weights.clone()
    }

    /// get the biases of the neurons in the layer
    pub fn get_biases(&self) -> ArrayD<f64> {
        self.biases.clone()
    }

    /// get the output of the neurons in the layer
    pub fn get_outputs(&self) -> Option<ArrayD<f64>> {
        self.outputs.clone()
    }

    fn forward(&mut self, inputs: ArrayD<f64>) {
        println!("{:?}", inputs.view());
        //self.outputs = Some(inputs.dot(&self.weights) + &self.biases)
    }

    fn backward(&mut self) {

    }
}


#[cfg(test)]
mod full_connected_layer_test {
    use super::FullConnectedLayer;


    #[test]
    fn two_neurons_weights_layer_shape() {
        // 4 inputs and 2 neurons for full connected neurons
        let layer = FullConnectedLayer::new(4, 2);
        let weights = layer.get_weights();

        assert_eq!(weights.shape(),  &[4,2]);
    }
}