use crate::neural_traits::LayerT;
use crate::activators::types::{Activator, ActivatorDeriv};
use crate::maths::Matrix;
use ndarray::{Array2, Axis};
use rand::random;

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
    pub fn new(n_inputs: usize, n_neurons: usize, activator: Activator, activator_deriv: Option<ActivatorDeriv>, layer_id: i32) -> FcLayer {
        
        
        // rows are for neurons and columns are for inputs
        let layer_weights = Array2::from_shape_fn((n_neurons, n_inputs), |(_, _)| random::<f64>());
        let weights = Matrix::new_from_array2(&layer_weights);

        
        // rows are neurons and each neurons has one bias
        let layer_biases = Array2::from_shape_fn((n_neurons, 1), |(_, _)| random::<f64>());
        let biases = Matrix::new_from_array2(&layer_biases);
    
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

        let wp: Array2<f64> = self.weights.get_data().dot(&inputs.get_data());
        
        let wp_b: Array2<f64> = wp + self.biases.get_data();
        let wp_b_mat = Matrix::new_from_array2(&wp_b);
        
        // save net_inputs
        self.net_inputs = Some(wp_b_mat.clone());

        // save input from previous layer
        self.inputs = Some(inputs.clone());

        // get the activation function
        let activ_func: fn(Matrix<f64>) -> Matrix<f64> = self.activator;

        // get layer output by apply the activation function
        let output = activ_func(wp_b_mat);
        Ok(output)
    }

    fn backward(&mut self, lr: f64, gradients: &Matrix<f64>) {

        // update Weights
        let inputs_trans = &self.get_inputs()
                                .unwrap()
                                .get_data().reversed_axes();

        let inputs_trans_mat = Matrix::new_from_array2(&inputs_trans);             
        let grad_inpt = gradients.get_data().dot(&inputs_trans_mat.get_data());
        let new_weights = self.weights.get_data() - (lr * grad_inpt);
        self.weights = Matrix::new_from_array2(&new_weights);

        // update Biases
        let nrows = gradients.get_nrows();
        let ncols = gradients.get_ncols();

        let grad = gradients
            .get_data()
            .sum_axis(Axis(1))
            .map(|a| *a / ncols as f64)
            .into_shape((nrows, 1)).unwrap();

        let new_bias = self.biases.get_data() - grad;
        self.biases = Matrix::new_from_array2(&new_bias);
        
    }

    /*fn save(&self) -> Option<Parameters> {
        Some(Parameters {
            layer_id: self.layer_id,
            layer_weights: self.weights.clone(),
            layer_biases: self.biases.clone()
        })

    }*/
    
    fn update_parameters(&mut self) {

    }

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

    fn get_inputs(&self) -> Option<Matrix<f64>> { self.inputs.clone()}
    
    fn get_net_inputs(&self) -> Option<Matrix<f64>> { self.net_inputs.clone()}

    fn get_biases(&self) -> Option<Matrix<f64>> { Some(self.biases.clone())}

    fn get_activator_deriv(&self) -> Option<ActivatorDeriv> {
        self.activator_deriv.clone()
    }

}
