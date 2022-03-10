use crate::{
    //loss_functions::{LossFunction, GradFunction},
    utils::{data_iter},
    neural_traits::{NetworkT},
    layers::multi_layers::{FcLayer},
    losses::Loss
};
//use serde_json::{to_writer, from_reader};
//use std::{io::BufReader, fs::File};
use crate::maths::Matrix; 
use ndarray::{Array2, Axis, ArrayBase};
use crate::activators::{Activation};



#[derive(Debug, Clone)]
pub struct FCNetwork {
    //network_db_id: String,
    network_inputs: Matrix<f64>,
    network_layers: Vec<FcLayer>,
    network_outputs: Matrix<f64>,
    loss: Loss
}


///
/// use ndarray::{Array2, Array};
/// use data::LottoClient;
/// use kongodjan::{
///    maths::Matrix,
///    network_arch::FCNetwork,
///    neural_traits::NetworkT,
///    activators::{Activation},
///    losses::Loss
/// };
///
///
/// fn main() {
///    // prepare data
///    let mut client = LottoClient::new();
///    
///    let (features, labels) = client.get_friday();
///
///    
///    let size = features.len();
///    let feat_nrows = features[0].len();
///    let lab_nrows = labels[0].len();
///    
///    let mut feat = Array2::zeros((feat_nrows, 0));
///    let mut lab = Array2::zeros((lab_nrows, 0));
///    
///    (0..size).into_iter().for_each(|a| {
///        feat.push_column(Array::from_vec(features[a].clone()).view()).unwrap();
///        lab.push_column(Array::from_vec(labels[a].clone()).view()).unwrap();
///    });
///    
///    let new_features = Matrix::new_from_array2(&feat);
///    let new_labels = Matrix::new_from_array2(&lab);
///
///    // build layers
///    let network_input = new_features.get_nrows();
///    let network_output_neurons = new_labels.get_nrows(); 
///    let hidden_layer_neurons = (((network_input + network_output_neurons) * 2) / 3) + 8;
///
///    // build the network
///    let mut network = FCNetwork::new(
///        new_features, 
///        &[
///            (network_input, None), 
///            (hidden_layer_neurons, Some(Activation::Sigmoid)), 
///            (hidden_layer_neurons, Some(Activation::Sigmoid)), 
///            (hidden_layer_neurons, Some(Activation::Sigmoid)), 
///            (network_output_neurons, Some(Activation::Sigmoid))
///        ],
///        new_labels,
///        Loss::Squared
///    );
///    
///    // train the networ
///    let batch_size: usize = 8;
///    let epoch = 500;
///    network.train(0.0003, batch_size, epoch); 
///    //192	monday special	27	4	2009	73	53	49	24	20	65	21	11	68	23
///    //193	monday special	4	5	2009	32	49	84	85	78	2	76	24	31	47
///
///    let pred_data = client.get_predict_input(&[73, 53, 49, 24, 20, 65, 21, 11, 68, 23]);
///    println!("test input: {:?}", pred_data);
///    let predicted = network.predict(&pred_data);
///    println!("tes output: {:?}", predicted.get_data());
///}
///

impl FCNetwork {
    pub fn new(network_inputs: Matrix<f64>, layers: &[(usize, Option<Activation>)], network_outputs: Matrix<f64>, loss: Loss) -> FCNetwork {
        
        let mut fcn = FCNetwork {
            //network_db_id: String::new(),
            network_inputs,
            network_layers: vec![],
            network_outputs,
            loss
        };
        
        for i in 1..layers.len() {
            
            let (n_inputs, _) = layers[i - 1];
            let (n_neurons, carrunt_act) = layers[i];
            let layer_id = i as i32;
            fcn.network_layers.push(FcLayer::new(n_inputs, n_neurons, carrunt_act.unwrap(), layer_id));
        }

        fcn
    }

    pub fn forword_propagation(&mut self, features: &Matrix<f64>) {

        let mut layer_output = features.clone();

        let forwod = |input: &Matrix<f64>, layer: &mut FcLayer | -> Matrix<f64> {
            // save input from previous layer
            layer.inputs = Some(input.clone());
            
            let wp: Array2<f64> = layer.weights.get_data().dot(&input.get_data());
            let mut wp_b: Array2<f64> = wp + layer.biases.get_data();
            layer.net_inputs = Some(Matrix::new_from_array2(&wp_b.clone()));
            
            wp_b.par_mapv_inplace(|d| {layer.activator.run(d)});
            let output = Matrix::new_from_array2(&wp_b);
            
            // get layer output by apply the activation function
            layer.outputs = Some(output.clone());
            output
        };


        for j in 0..self.network_layers.len(){
            layer_output = forwod(&layer_output, &mut self.network_layers[j]);
        }
    }

    pub fn backword_propagation(&mut self, err_deriv: &Matrix<f64>) {

        self.network_layers.reverse();

        let mut prev_gradient = err_deriv.clone();

        let backword = |der: &Matrix<f64>, layer: &mut FcLayer| -> Matrix<f64> {
            // net_inputs_deriv
            let mut net_inputs_deriv = layer.net_inputs.clone().unwrap().get_data();
            net_inputs_deriv.par_mapv_inplace(|d| {layer.activator.derivative(d)});
            
            // local gradient
            layer.local_gradient = Some(Matrix::new_from_array2(&(der.get_data() * net_inputs_deriv.clone())));
            
            // gradient for previous layer
            let weights_t = layer.weights.get_data().clone().reversed_axes();
            let grad = weights_t.dot(&net_inputs_deriv);

            Matrix::new_from_array2(&grad)
        };

        for j in 0..self.network_layers.len(){
            prev_gradient = backword(&prev_gradient, &mut self.network_layers[j]);
        }
        
        self.network_layers.reverse();
    }

    pub fn update_parameters(&mut self, lr: f64) {

        for i in 0..self.network_layers.len() {
            
            let biases = self.network_layers[i].biases.clone().get_data();
            let weights = self.network_layers[i].weights.clone().get_data();
            let inputs = self.network_layers[i].inputs.clone().unwrap().get_data().reversed_axes();
            let local_grad = self.network_layers[i].local_gradient.clone().unwrap().get_data();
            let local_grad_b = local_grad.clone().sum_axis(Axis(1));
            
            let local_grad_inputs = local_grad.dot(&inputs);

            let lr_arr = Array2::from_shape_fn((weights.nrows(), weights.ncols()), |_| lr);
            let lr_arr_local_grad_inputs = lr_arr * local_grad_inputs;
            
            self.network_layers[i].weights = 
                Matrix::new_from_array2(&(weights - lr_arr_local_grad_inputs)); 

            // biases 
            let mut local_grad_b_arr = local_grad_b.into_shape((biases.nrows(), 1)).unwrap();
            local_grad_b_arr.par_mapv_inplace(|d| {d / biases.ncols() as f64});

            let lr_arr_b = Array2::from_shape_fn((biases.nrows(), 1), |_| lr);
            let lr_arr_b_local_grad_b_arr = lr_arr_b * local_grad_b_arr;
            self.network_layers[i].biases = 
                Matrix::new_from_array2(&(biases - lr_arr_b_local_grad_b_arr)); 

        }
        
    }

    pub fn pridict(&self) {

    }
}

impl NetworkT for FCNetwork {
    fn train(&mut self, lr: f64, batch_size: usize, epoch: i32) {

        
        //let mut round: i64 = 0;
        for round in 0..epoch {
            for (feature, label) in data_iter(batch_size, &self.network_inputs, &self.network_outputs) {
                
                self.forword_propagation(&feature);
                let output = self.network_layers.last().unwrap().outputs.clone().unwrap();
                //let loss = self.loss.run(&output, &label);
                //println!("Loss: {:?}", loss);
                let err_deriv = self.loss.derivative(&output, &label);
                self.backword_propagation(&err_deriv);
                self.update_parameters(lr);
            } 
            
            self.forword_propagation(&self.network_inputs.clone());
            let output = self.network_layers.last().unwrap().outputs.clone().unwrap();
            let loss = self.loss.run(&output, &self.network_outputs);
            println!("Epoch: {:?} Loss: {:?}", round, loss);
            println!(" ");
            let err_deriv = self.loss.derivative(&output, &self.network_outputs);
            self.backword_propagation(&err_deriv);
            self.update_parameters(lr);
            
        }
        
    }

    fn predict(&mut self, input: &[f64]) -> Matrix<f64> {

        let input = ArrayBase::from_vec(input.to_vec()).into_shape((input.len(), 1)).unwrap();
        self.forword_propagation(&Matrix::new_from_array2(&input));

        Matrix::new_from_array2(
            &ArrayBase::from_vec(
                self.network_layers
                .last()
                .unwrap()
                .outputs
                .clone()
                .unwrap()
                .get_data()
                .column(0)
                .to_vec())
            .into_shape((10, 24))
            .unwrap()
        )
    }

}
