use ndarray::{ArrayD, IxDyn, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use crate::neural_traits::LayerT;
use crate::activations::{Activator};


#[derive(Debug, Clone)]
pub struct FullConnectedLayer {
    weights: ArrayD<f64>,
    biases: ArrayD<f64>,
    activator: Activator
}

impl FullConnectedLayer {
    /// create a new full connected layer  
    /// n_inputs is the number of rows in the input matrix    
    /// n_neurons is the number of neurons in the layer  
    pub fn new(n_inputs: usize, n_neurons: usize, f: Activator) -> Self {
        let weights = ArrayD::random(IxDyn(&[n_inputs, n_neurons]), Uniform::new(-1., 1.));
        let biases = ArrayD::random(IxDyn(&[1, n_neurons]), Uniform::new(-1., 1.));
        FullConnectedLayer {
            weights,
            biases,
            activator: f
        }
    }
}

impl LayerT for FullConnectedLayer {

    fn get_weights(&self) -> ArrayD<f64> {
        self.weights.clone()
    }

    fn get_biases(&self) -> ArrayD<f64> {
        self.biases.clone()
    }

    fn forward(&self, inputs: ArrayD<f64>) -> ArrayD<f64> {

        let mut net_input_vec = Vec::new();

        let weights_row = self.weights.shape()[1];

        let mut weights_vec = Vec::new();
        
        for row in self.weights.genrows() {

            weights_vec.push(row);
        };

        if let &[i ,j] = self.weights.shape() {
            for j_val in 0..j {
                // neurons here
                let mut sum = 0.0;
                let bias = self.biases.as_slice().unwrap()[j_val];

                for i_val in 0..i {
                    let w = weights_vec[i_val][j_val];
                    let p = inputs.as_slice().unwrap()[i_val];
                    sum = sum + (w*p);
                }
                
                net_input_vec.push(sum + bias);

            }
        }
        
        // use the activation function
        let activ: fn(f64) -> f64 = self.activator;

        let outputs = Array::from_shape_fn(IxDyn(&[weights_row, 1]), |args| {
            
            let arg = args[0];
            let value = net_input_vec[arg];
            
            activ(value)
        });

        outputs
        
    }

    fn backward(&mut self, inputs: ArrayD<f64>) {
        println!("{:?}", inputs.view());
    }
}

#[cfg(test)]
mod full_connected_layer_test {
    use super::FullConnectedLayer;
    use crate::neural_traits::LayerT;
    use crate::activations::tansig;

    #[test]
    fn two_neurons_weights_layer_shape() 
    {

        // 4 inputs and 2 neurons for full connected neurons
        let layer = FullConnectedLayer::new(4, 2, tansig);
        let weights = layer.get_weights();

        assert_eq!(weights.shape(),  &[4,2]);
    }
}