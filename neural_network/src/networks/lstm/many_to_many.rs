use crate::{
    layers::{LossLayer, LstmLayer},
    //activator::Activator,
    loss::Loss,
    //networks::NetworkT,
    //utils::data_iter,
    //io::IO
};
use algo_diff::{
    maths::{Matrix, Axis},
    //graphs::{DotGraph, Graph}
};
use rand::random;
//use std::collections::HashMap;



#[derive(Debug)]
pub struct LstmManyToMany {
    parameters: Matrix,
    layer: LstmLayer,
    outputs: Vec<Matrix>,
    loss: LossLayer,
    n_neurons: usize
}

impl LstmManyToMany {
    pub fn new(n_inputs: usize, n_neurons: usize, loss: Loss) -> LstmManyToMany {
        
        // forget_gate
            // inputs weights
            let f_inputs_weights = Matrix::from_shape_fn((n_inputs, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // hidden weights
            let f_hidden_weights = Matrix::from_shape_fn((n_neurons, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // biases
            let f_biases = Matrix::from_shape_fn((1, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
        // input_gate
            // inputs weights
            let i_inputs_weights = Matrix::from_shape_fn((n_inputs, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // hidden weights
            let i_hidden_weights = Matrix::from_shape_fn((n_neurons, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // biases
            let i_biases = Matrix::from_shape_fn((1, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
        // output_gate
            // inputs weights
            let o_inputs_weights = Matrix::from_shape_fn((n_inputs, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // hidden weights
            let o_hidden_weights = Matrix::from_shape_fn((n_neurons, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // biases
            let o_biases = Matrix::from_shape_fn((1, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
        // candidate
            // inputs weights
            let c_inputs_weights = Matrix::from_shape_fn((n_inputs, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // hidden weights
            let c_hidden_weights = Matrix::from_shape_fn((n_neurons, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
            // biases
            let c_biases = Matrix::from_shape_fn((1, n_neurons), |_| 2f64 * random::<f64>() - 1f64);

        let mut inputs_weights = Matrix::zeros((n_inputs,0));
        inputs_weights.append(Axis(1), f_inputs_weights.view()).unwrap();
        inputs_weights.append(Axis(1), i_inputs_weights.view()).unwrap();
        inputs_weights.append(Axis(1), o_inputs_weights.view()).unwrap();
        inputs_weights.append(Axis(1), c_inputs_weights.view()).unwrap();

        let mut hidden_weights = Matrix::zeros((n_neurons,0));
        hidden_weights.append(Axis(1), f_hidden_weights.view()).unwrap();
        hidden_weights.append(Axis(1), i_hidden_weights.view()).unwrap();
        hidden_weights.append(Axis(1), o_hidden_weights.view()).unwrap();
        hidden_weights.append(Axis(1), c_hidden_weights.view()).unwrap();
        
        let mut biases = Matrix::zeros((1,0));
        biases.append(Axis(1), f_biases.view()).unwrap();
        biases.append(Axis(1), i_biases.view()).unwrap();
        biases.append(Axis(1), o_biases.view()).unwrap();
        biases.append(Axis(1), c_biases.view()).unwrap();

        let mut parameters = Matrix::zeros((0, 4 * n_neurons));
        parameters.append(Axis(0), inputs_weights.view()).unwrap();
        parameters.append(Axis(0), hidden_weights.view()).unwrap();
        parameters.append(Axis(0), biases.view()).unwrap();

        parameters = parameters.reversed_axes();


        let los = LossLayer::new(loss.clone());
        LstmManyToMany {
            parameters,
            layer: LstmLayer::new(),
            outputs: vec![],
            loss: los,
            n_neurons
        }
    }

    pub fn train(&mut self, inputs: &[Vec<Matrix>]) {
        
        let mut hidden_state: Matrix = Matrix::zeros((self.n_neurons, 1));
        let mut memory_state: Matrix = Matrix::zeros((self.n_neurons, 1));
        let mut gradient_vec: Vec<Matrix> = vec![];
        //println!("parameters: {:#?}", self.parameters.shape());
        // Forward
        for input in inputs[0].clone() {
            let mut gradients: Matrix = Matrix::zeros((self.n_neurons, 1));
            (hidden_state, memory_state, gradients) = self.layer.forward(&self.parameters, input, hidden_state, memory_state);
            //println!("gradients: {:#?}", gradients.shape());
            gradient_vec.push(gradients);
        }

        // Outputs 

        // Loss Layer

        // Backpropagation

        // update parameters
    }
}