use ndarray::{ArrayD, IxDyn, Array};
use ndarray_rand::rand::{thread_rng, Rng};
use crate::neural_traits::LayerT;
use crate::activations::{Activator, ActivatorDeriv, FLOAT_SIZE_10000};
use crate::errors::Gradient;


#[derive(Debug, Clone)]
pub struct FullConnectedLayer {
    inputs: Option<ArrayD<f64>>,
    net_inputs: Option<ArrayD<f64>>,
    weights: ArrayD<f64>,
    biases: ArrayD<f64>,
    activator: Activator,
    activator_deriv: ActivatorDeriv,
}

impl FullConnectedLayer {
    /// create a new full connected layer  
    /// n_inputs is the number of rows in the input matrix    
    /// n_neurons is the number of neurons in the layer  
    pub fn new(n_inputs: usize, n_neurons: usize, activator: Activator, activator_deriv: ActivatorDeriv) -> Self {
        let mut rng = thread_rng(); 
        let weights = Array::from_shape_fn(IxDyn(&[n_inputs, n_neurons]), |_args| {
            
            let value = ((rng.gen::<f64>() * FLOAT_SIZE_10000).trunc())/FLOAT_SIZE_10000;
            value
        });

        let biases = Array::from_shape_fn(IxDyn(&[1, n_neurons]), |_args| {
            
            let value = ((rng.gen::<f64>() * FLOAT_SIZE_10000).trunc())/FLOAT_SIZE_10000;
            value
        });

        FullConnectedLayer {
            inputs: None,
            net_inputs: None,
            weights,
            biases,
            activator,
            activator_deriv
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

    fn forward(&mut self, inputs: ArrayD<f64>) -> ArrayD<f64> {
        let mut net_input_vec = Vec::new();

        self.inputs = Some(inputs.clone());

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
                    sum = ((sum + (w*p)) * 1000.0).trunc() / 1000.0;
                }
                net_input_vec.push(sum + bias);

            }
        }
        
        // use the activation function
        let activ: fn(f64) -> f64 = self.activator;
        
        let outputs = Array::from_shape_fn(IxDyn(&[weights_row, 1]), |args| {
            
            let arg = args[0];
            let value = net_input_vec[arg];
            
            let res = activ(value);
            res
        });

        let net_inputs = Array::from_shape_fn(IxDyn(&[weights_row, 1]), |args| {
            
            let arg = args[0];
            net_input_vec[arg]
        });

        self.net_inputs = Some(net_inputs);

        outputs
        
    }

    fn backward(&mut self, error_der: ArrayD<f64>, rate: f64, gradient: Gradient) -> ArrayD<f64> {

        let gradient_func: fn(deri: f64, error: f64) -> f64 = gradient;

        let neur_layer_weights = self.get_weights();
        let input_size = neur_layer_weights.shape()[0];
        let neur_size = neur_layer_weights.shape()[1];
        let layer_inputs = self.inputs.clone().unwrap();
        let layer_inp = layer_inputs.as_slice().unwrap();
        let act_der_fun: fn(f64) -> f64 = self.activator_deriv;
        //println!("old weights: {:?}", self.weights);
        let mut weights_vec = Vec::new();
        
        for row in self.weights.genrows() {

            weights_vec.push(row);
        };

        let mut gradient_vec = Vec::new();
        let mut new_layer_weights_vec = Vec::new();
        let mut new_layer_bias_vec = Vec::new();
        // let go through each neuron in th layer
        if let Some(net_input) = &self.net_inputs {
            for n_i in 0..neur_size {
                
                // neurons levels
                let er_de = error_der.as_slice().unwrap()[n_i];
                let net_inp = net_input.as_slice().unwrap()[n_i];
                
                let acti_derive =  act_der_fun(net_inp); 

                let gradient = gradient_func(acti_derive, er_de);


                let mut neuron_weight_vec = Vec::new();

                let bias = self.biases.as_slice().unwrap()[n_i];

                for r_i in 0..input_size {
                    // input and weights levels
                    let w = weights_vec[r_i][n_i];
                    let input = layer_inp[r_i];                  
                    let new_w = w - (rate * gradient * input);
                    let new_w_val = (new_w * FLOAT_SIZE_10000).trunc() / FLOAT_SIZE_10000;
                    neuron_weight_vec.push(new_w_val);
                    
                }
                
                let new_bias = bias - (rate * gradient);
                let new_bias_val = (new_bias * FLOAT_SIZE_10000).trunc() / FLOAT_SIZE_10000;
                gradient_vec.push(gradient);
                new_layer_weights_vec.push(neuron_weight_vec);
                new_layer_bias_vec.push(new_bias_val);

            }
        } 

        // get error derive for back layer
        let mut error_derivation_vec =  Vec::new();

        for r_i in 0..input_size {

            // neuron level
            let mut gradient_sum = 0.0;
            for n_i in 0..neur_size {
                let w = weights_vec[r_i][n_i];
                let g = gradient_vec[n_i];
                gradient_sum = gradient_sum + (w * g);
            }
            error_derivation_vec.push(gradient_sum);
        }
        
        //println!("weights: {:?}", new_layer_weights_vec[0].len());
        let r = new_layer_weights_vec[0].len();
        let n = new_layer_weights_vec.len();

        let new_layer_weights = Array::from_shape_fn(IxDyn(&[r, n]), |args| {
            
            let a = args[0];
            let b = args[1];
            new_layer_weights_vec[b][a]
        });

        self.weights = new_layer_weights;

        let new_layer_bias = Array::from_shape_fn(IxDyn(&[1, n]), |args| {
            
            let a = args[1];
            new_layer_bias_vec[a]
        });

        self.biases = new_layer_bias;
        
        let outputs = Array::from_shape_fn(IxDyn(&[input_size, 1]), |args| {
            
            let arg = args[0];
            error_derivation_vec[arg]
        });

        outputs
    }

    
}

#[cfg(test)]
mod full_connected_layer_test {
    use super::FullConnectedLayer;
    use crate::neural_traits::LayerT;
    use crate::activations::{tansig, tansig_deriv};

    #[test]
    fn two_neurons_weights_layer_shape() 
    {

        // 4 inputs and 2 neurons for full connected neurons
        let layer = FullConnectedLayer::new(4, 2, tansig, tansig_deriv);
        let weights = layer.get_weights();

        assert_eq!(weights.shape(),  &[4,2]);
    }
}