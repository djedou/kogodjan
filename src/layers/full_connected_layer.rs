use crate::{
    neural_traits::LayerT,
    activators::{Activation}
};
use ndarray::{Array2};
use rand::random;



/// Linear Regression Layer
#[derive(Debug, Clone)]
pub struct FcLayer {
    pub layer_id: i32,
    pub inputs: Option<Vec<Array2<f64>>>,
    pub net_inputs: Option<Vec<Array2<f64>>>,
    pub outputs: Option<Vec<Array2<f64>>>,
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub activator: Activation,
    pub local_gradient: Option<Vec<Array2<f64>>>,
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
    fn forword(&mut self, input: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        // save input from previous layer
        self.inputs = Some(input.clone().to_owned());
        
        let mut net_inputs_vec: Vec<Array2<f64>> = vec![];
        let mut outputs_vec: Vec<Array2<f64>> = vec![];

        let local_forword = |inp: &Array2<f64>, weights: Array2<f64>, bias: Array2<f64>, act: &Activation| -> (Array2<f64>, Array2<f64>) {
            let wp: Array2<f64> = weights.dot(inp);
            let mut wp_b: Array2<f64> = wp + bias;
            let net = wp_b.clone();
            
            // apply the layer activation
            wp_b.par_mapv_inplace(|d| {act.run(d)});
            
            (net, wp_b)
        };

        for inp in input {

            let (net, out) = local_forword(&inp, self.weights.clone(), self.biases.clone(), &self.activator);
            net_inputs_vec.push(net);
            outputs_vec.push(out);
        }

        self.net_inputs = Some(net_inputs_vec.clone());
        self.outputs = Some(outputs_vec.clone());
        
        outputs_vec
    }

    fn backword(&mut self, gradient: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        let mut local_gradients: Vec<Array2<f64>> = vec![];
        let mut previous_gradients: Vec<Array2<f64>> = vec![];

        let back = |grad: &Array2<f64>, net: Array2<f64>, weights: Array2<f64>, act: &Activation| -> (Array2<f64>, Array2<f64>) {
            // net_inputs_deriv
            let mut net_inputs_deriv = net.clone();
            net_inputs_deriv.par_mapv_inplace(|d| {act.derivative(d)});
            
            // local gradient
            let local = grad * net_inputs_deriv;
            
            // gradient for previous layer
            let previous = weights.dot(&net); // return the new gradient
            
            (local, previous)
        };

        for (i, grad) in gradient.iter().enumerate() {
            let (local, previous) = back(&grad, self.net_inputs.clone().unwrap()[i].clone(), self.weights.clone().reversed_axes(), &self.activator);
            local_gradients.push(local);
            previous_gradients.push(previous);
        }

        self.local_gradient = Some(local_gradients);

        previous_gradients
    }

    fn update_parameters(&mut self, lr: f64) {
        let local_grad = self.local_gradient.clone().unwrap();
        let inputs = self.inputs.clone().unwrap();

        let update = |lr_w: Array2<f64>, lr_b: Array2<f64>, grad: Array2<f64>, inp: Array2<f64>| -> (Array2<f64>, Array2<f64>) {

            let upd_weights = lr_w * (grad.dot(&inp));
            let upd_bias = lr_b * grad;

            (upd_weights, upd_bias)
        };
        
        let lr_arr_b = Array2::from_shape_fn((self.biases.nrows(), 1), |_| lr);
        let lr_arr_w = Array2::from_shape_fn((self.weights.nrows(), self.weights.ncols()), |_| lr);

        for (i, grad) in local_grad.iter().enumerate() {
            let (upd_weights, upd_bias) = update(lr_arr_w.clone(), lr_arr_b.clone(), grad.clone(), inputs[i].clone().reversed_axes());
            
            self.weights = self.weights.clone() - upd_weights;
            self.biases = self.biases.clone() - upd_bias;
        }
    }

    fn predict_forword(&mut self, input: Vec<Array2<f64>>) -> Vec<Array2<f64>> {
        let mut pred_res: Vec<Array2<f64>> = vec![];

        let pred = |inp: &Array2<f64>, weights: Array2<f64>, bias: Array2<f64>, act: &Activation| -> Array2<f64> {
            let wp: Array2<f64> = weights.dot(inp);
            let mut wp_b: Array2<f64> = wp + bias;
            
            // apply the layer activation
            wp_b.par_mapv_inplace(|d| {act.run(d)});
            
            wp_b
        };

        for inp in input {
            let out = pred(&inp, self.weights.clone(), self.biases.clone(), &self.activator);
            pred_res.push(out);
        }

        self.outputs = Some(pred_res.clone());
        pred_res
    }
}