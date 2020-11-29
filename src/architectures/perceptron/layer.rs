use ndarray::{ArrayD, IxDyn};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::neural_traits::LayerT;

#[derive(Debug, Clone)]
pub struct FullConnectedLayer<F> {
    weights: ArrayD<f64>,
    biases: ArrayD<f64>,
    outputs: Option<ArrayD<f64>>,
    activator: Option<F>
}

impl<F> FullConnectedLayer<F> {
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
            activator: None
        }
    }

    fn forward(&mut self, inputs: ArrayD<f64>) {
        println!("{:?}", inputs.view());
        //self.outputs = Some(inputs.dot(&self.weights) + &self.biases)

        //let activator_input = inputs.dot(&self.weights) + &self.biases;

    }

    fn backward(&mut self) {

    }
}

impl<F> LayerT for FullConnectedLayer<F> {

    type Weights = ArrayD<f64>;
    type Biases = ArrayD<f64>;
    type Outputs = ArrayD<f64>;
 
    fn get_weights(&self) -> Self::Weights {
        self.weights.clone()
    }

    fn get_biases(&self) -> Self::Biases {
        self.biases.clone()
    }

    fn get_outputs(&self) -> Option<Self::Outputs> {
        self.outputs.clone()
    }

}


#[cfg(test)]
mod full_connected_layer_test {
    use super::FullConnectedLayer;
    use ndarray::{ArrayD};
    use crate::neural_traits::LayerT;

    #[test]
    fn two_neurons_weights_layer_shape() 
    {
        // 4 inputs and 2 neurons for full connected neurons
        let layer = FullConnectedLayer::<Box<dyn Fn(ArrayD<f64>) -> ArrayD<f64>>>::new(4, 2);
        let weights = layer.get_weights();

        assert_eq!(weights.shape(),  &[4,2]);
    }
}