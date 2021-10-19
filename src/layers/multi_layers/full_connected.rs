//use random_number::rand::{thread_rng, Rng};
use crate::neural_traits::LayerT;
use crate::activators::types::{Activator, ActivatorDeriv};
use std::sync::{Arc, Mutex};
use crate::{
    optimizers::Optimizer
};
//use djed_maths::linear_algebra::vector::{ZipCallBack};

use djed_maths::linear_algebra::matrix::Matrix;
/// Linear Regression Layer
#[derive(Debug, Clone)]
pub struct FcLayer {
    layer_id: i32,
    inputs: Option<Matrix<f64>>,
    net_inputs: Option<Matrix<f64>>,
    weights: Matrix<f64>,
    biases: Matrix<f64>,
    activator: Activator, 
    activator_deriv: Option<ActivatorDeriv>
}

impl FcLayer {
    /// create a new Full Connected Layer  
    /// n_inputs(neuron input) is the number of input for the single neuron in the layer 
    /// n_neurons is the number of neurons for the layer    
    pub fn new(n_inputs: usize, n_neurons: usize, batch_size: usize, activator: Activator, activator_deriv: Option<ActivatorDeriv>, layer_id: i32) -> FcLayer {
        
        let weights_callback = |_x| {
            //let value: f64 = random!(); // generate float between 0.0 and 1.0
            //value
            random!()
        };
        // rows are for neurons and columns are for inputs
        let weights = Matrix::new_from_fn(n_neurons, n_inputs, weights_callback);
        
        let biases_callback = |_x| {
            //let value: f64 = random!(); // generate float between 0.0 and 1.0
            //value
            random!()
        };
 
        let biases = Matrix::new_from_fn(n_neurons, batch_size, biases_callback);
    
        FcLayer {
            layer_id,
            inputs: None,
            net_inputs: None,
            weights,
            biases,
            activator,
            activator_deriv

        }
    }
}



impl LayerT for FcLayer {

    fn forward(&mut self, inputs: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        // save input from previous layer
        self.inputs = Some(inputs.clone());

        // get the activation function
        let activ_func: fn(Matrix<f64>) -> Matrix<f64> = self.activator;

        // calcule weights_inputs = weights * inputs
        let mut wx = self.weights.dot_product(&inputs).unwrap();
        
        // prepared bias
        let new_bias = self.biases.clone();
        
        // wx + b
        let wx_b = wx.add_matrix(&new_bias).unwrap();
        
        // save net_inputs
        self.net_inputs = Some(wx_b.clone());

        // get layer output by apply the activation function
        let output = activ_func(wx_b);
        Ok(output)
    }

    fn backward(&mut self, lr: f64, gradients: &Matrix<f64>, optimizer: &Optimizer) -> Matrix<f64> {

        let opt_func: fn(lr: f64, gradient: &Matrix<f64>, param: &Matrix<f64>,  input: Option<&Matrix<f64>>) -> Matrix<f64> = *optimizer;

        let new_grad = self.weights.transpose().dot_product(&gradients).unwrap();

        let deriv: fn(Matrix<f64>) -> Matrix<f64> = if let Some(d) = self.activator_deriv {
            d
        } else {
            panic!("please provide derivative for all activators");
        };
        
        // The net_input gradient
        let layer_grad = if let Some(ref net) = self.net_inputs {
            deriv(net.clone())
        } else {
            panic!("this layer can not be used");
        };

        let callback_grad = |a,b| a * b;
        let grad_driv: Matrix<f64> = gradients.zip_apply(&layer_grad, callback_grad).unwrap();

        self.weights = opt_func(lr, &grad_driv, &self.weights, Some(&self.inputs.clone().unwrap()));
        self.biases = opt_func(lr, &grad_driv, &self.biases, None);

        new_grad
        
    }

    /*fn save(&self) -> Option<Parameters> {
        Some(Parameters {
            layer_id: self.layer_id,
            layer_weights: self.weights.clone(),
            layer_biases: self.biases.clone()
        })

    }*/
    
    fn get_layer_id(&self) -> i32 {
        self.layer_id
    }

    fn set_weights(&mut self, weights: Matrix<f64>) {
        self.weights = weights;
    }

    fn set_biases(&mut self, biases: Matrix<f64>) {
        self.biases = biases;
    }

    fn get_weights(&self) -> Option<Matrix<f64>> { Some(self.weights.clone())}

    fn get_biases(&self) -> Option<Matrix<f64>> { Some(self.biases.clone())}

    fn get_activator_deriv(&self) -> Option<ActivatorDeriv> {
        self.activator_deriv.clone()
    }

}
