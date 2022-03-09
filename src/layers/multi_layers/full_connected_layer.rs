use crate::activators::{Activation};
use crate::maths::Matrix;
use ndarray::{Array2};
use rand::random;

/// Linear Regression Layer
#[derive(Debug, Clone)]
pub struct FcLayer {
    pub layer_id: i32,
    pub inputs: Option<Matrix<f64>>,
    pub net_inputs: Option<Matrix<f64>>,
    pub outputs: Option<Matrix<f64>>,
    pub weights: Matrix<f64>,
    pub biases: Matrix<f64>,
    pub activator: Activation,
    pub local_gradient: Option<Matrix<f64>>,
}

impl FcLayer {
    /// create a new Full Connected Layer  
    /// n_inputs(neuron input) is the number of input for the single neuron in the layer 
    /// n_neurons is the number of neurons for the layer    
    pub fn new(n_inputs: usize, n_neurons: usize, activator: Activation, layer_id: i32) -> FcLayer {
        
        

        // rows are for neurons and columns are for inputs
        let layer_weights = Array2::from_shape_fn((n_neurons, n_inputs), |_| 2f64 * random::<f64>() - 1f64);
        let weights = Matrix::new_from_array2(&layer_weights);

        
        // rows are neurons and each neurons has one bias
        let layer_biases = Array2::from_shape_fn((n_neurons, 1), |_| 2f64 * random::<f64>() - 1f64);
        let biases = Matrix::new_from_array2(&layer_biases);
    
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
