use crate::{
    neural_traits::LayerT,
    activators::{Activation}
};
use ndarray::{Array2, Axis};
use rand::random;



/// Linear Regression Layer
#[derive(Debug, Clone)]
pub struct FcLayer {
    pub layer_id: i32,
    pub inputs: Option<Array2<f64>>,
    pub net_inputs: Option<Array2<f64>>,
    pub outputs: Option<Array2<f64>>,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activator: Activation,
    pub local_gradient: Option<Array2<f64>>,
}

impl FcLayer {
    /// create a new Full Connected Layer  
    /// n_inputs(neuron input) is the number of input for the single neuron in the layer 
    /// n_neurons is the number of neurons for the layer    
    pub fn new(n_inputs: usize, n_neurons: usize, activator: Activation, layer_id: i32) -> FcLayer {
        // rows are for neurons and columns are for inputs
        let weights = Array2::from_shape_fn((n_neurons, n_inputs), |_| 2f64 * random::<f64>() - 1f64);
        
        // rows are neurons and each neurons has one bias
        let biases = Array2::from_shape_fn((n_neurons, 1), |_| 2f64 * random::<f64>() - 1f64);
    
        FcLayer {
            layer_id,
            inputs: None,
            net_inputs: None,
            outputs: None,
            weights,
            biases,
            activator,
            local_gradient: None
        }
    }
}


impl LayerT for FcLayer {
    fn forword(&mut self, input: &Array2<f64>) -> Array2<f64> {
        // save input from previous layer
        self.inputs = Some(input.clone());
        
        let wp: Array2<f64> = self.weights.dot(input);
        let mut wp_b: Array2<f64> = wp + self.biases.clone();
        self.net_inputs = Some(wp_b.clone());
        
        // apply the layer activation
        wp_b.par_mapv_inplace(|d| {self.activator.run(d)});
        
        self.outputs = Some(wp_b.clone());
        wp_b
    }

    fn backword(&mut self, gradient: &Array2<f64>) -> Array2<f64> {
        // net_inputs_deriv
        let mut net_inputs_deriv = self.net_inputs.clone().unwrap();
        net_inputs_deriv.par_mapv_inplace(|d| {self.activator.derivative(d)});
        
        // local gradient
        self.local_gradient = Some(gradient * net_inputs_deriv.clone());
        
        // gradient for previous layer
        let weights_t = self.weights.clone().reversed_axes(); // transpose the weights
        weights_t.dot(&net_inputs_deriv) // return the new gradient
    }

    fn update_parameters(&mut self, lr: f64) {
        let inputs = self.inputs.clone().unwrap().reversed_axes();
        let local_grad = self.local_gradient.clone().unwrap();
        let local_grad_b = local_grad.clone().sum_axis(Axis(1));

        let lr_arr = Array2::from_shape_fn((self.weights.nrows(), self.weights.ncols()), |_| lr);
        let lr_arr_local_grad_inputs = lr_arr * (local_grad.dot(&inputs));
        
        self.weights = self.weights.clone() - lr_arr_local_grad_inputs; 

        // biases 
        let mut local_grad_b_arr = local_grad_b.into_shape((self.biases.nrows(), 1)).unwrap();
        local_grad_b_arr.par_mapv_inplace(|d| {d / self.biases.ncols() as f64});

        let lr_arr_b = Array2::from_shape_fn((self.biases.nrows(), 1), |_| lr);
        let lr_arr_b_local_grad_b_arr = lr_arr_b * local_grad_b_arr;

        self.biases = self.biases.clone() - lr_arr_b_local_grad_b_arr; 
    }

    fn predict_forword(&mut self, input: &Array2<f64>) -> Array2<f64> {
        let wp: Array2<f64> = self.weights.dot(input);
        let mut wp_b: Array2<f64> = wp + self.biases.clone();
        
        // apply the layer activation
        wp_b.par_mapv_inplace(|d| {self.activator.run(d)});
        
        self.outputs = Some(wp_b.clone());
        wp_b
    }
}