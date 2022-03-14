use crate::{
    utils::{data_iter},
    neural_traits::{NetworkT, LayerT},
    layers::{FcLayer},
    losses::Loss
};
//use serde_json::{to_writer, from_reader};
//use std::{io::BufReader, fs::File};
use ndarray::{Array2, ArrayBase};
use crate::activators::{Activation};



///
/// use ndarray::{Array2, Array};
/// use data::LottoClient;
/// use kongodjan::{
///    maths::Array2,
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
///    let mut new_features = Array2::zeros((feat_nrows, 0));
///    let mut new_labels = Array2::zeros((lab_nrows, 0));
///    
///    (0..size).into_iter().for_each(|a| {
///        new_features.push_column(Array::from_vec(features[a].clone()).view()).unwrap();
///        new_labels.push_column(Array::from_vec(labels[a].clone()).view()).unwrap();
///    });
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
#[derive(Debug, Clone)]
pub struct FCNetwork {
    //network_db_id: String,
    network_inputs: Vec<Array2<f64>>,
    network_layers: Vec<FcLayer>,
    network_outputs: Vec<Array2<f64>>,
    loss: Loss,
    network_erros: Vec<f64>
}


impl FCNetwork {
    pub fn new(network_inputs: Vec<Array2<f64>>, layers: &[(usize, Option<Activation>)], network_outputs: Vec<Array2<f64>>, loss: Loss) -> FCNetwork {
        
        let mut fcn = FCNetwork {
            //network_db_id: String::new(),
            network_inputs,
            network_layers: vec![],
            network_outputs,
            loss,
            network_erros: vec![]
        };
        
        for i in 1..layers.len() {
            
            let (n_inputs, _) = layers[i - 1];
            let (n_neurons, carrunt_act) = layers[i];
            let layer_id = i as i32;
            fcn.network_layers.push(FcLayer::new(n_inputs, n_neurons, carrunt_act.unwrap(), layer_id));
        }

        fcn
    }

    pub fn get_errors(&self) -> Vec<f64> {
        self.network_erros.clone()
    }
}



impl NetworkT for FCNetwork {
    fn train(&mut self, lr: f64, batch_size: usize, epoch: i32) {
        //let mut round: i64 = 0;
        for round in 0..epoch {
            for (feature, label) in data_iter(batch_size, &self.network_inputs, &self.network_outputs) {
                self.forword_propagation(feature);
                let output_layer = self.network_layers.last().unwrap();
                let err_deriv = self.loss.derivative(output_layer.outputs.clone().unwrap(), &label);
                self.backword_propagation(err_deriv);
                self.update_parameters(lr);
            } 
            
            self.forword_propagation(self.network_inputs.clone());
            let output = self.network_layers.last().unwrap().outputs.clone().unwrap();
            //self.network_erros = self.loss.run(&output, &self.network_outputs);
            let network_erros = self.loss.run(&output, &self.network_outputs);
            println!("Epoch: {:?} Errors: {:?}", round, network_erros);
            let err_deriv = self.loss.derivative(output, &self.network_outputs);
            self.backword_propagation(err_deriv);
            self.update_parameters(lr);
        }
    }

    fn forword_propagation(&mut self, features: Vec<Array2<f64>>) {
        let mut layer_output = features.clone();

        for j in 0..self.network_layers.len(){
            layer_output = self.network_layers[j].forword(layer_output);
        }
    }

    fn predict_forword_propagation(&mut self, features: Vec<Array2<f64>>) {
        let mut layer_output = features.clone();

        for j in 0..self.network_layers.len(){
            layer_output = self.network_layers[j].predict_forword(layer_output);
        }
    }

    fn backword_propagation(&mut self, gradient: Vec<Array2<f64>>) {

        self.network_layers.reverse();
        let mut prev_gradient = gradient.clone();

        for j in 0..self.network_layers.len(){
            prev_gradient = self.network_layers[j].backword(prev_gradient);
        }
        
        self.network_layers.reverse();
    }

    fn update_parameters(&mut self, lr: f64) {

        for i in 0..self.network_layers.len() {
            self.network_layers[i].update_parameters(lr); 
        }
    }

    fn network_outpout(&self) -> Vec<Array2<f64>> {

        self.network_layers
            .last()
            .unwrap()
            .clone()
            .outputs
            .unwrap()
    }

    fn predict(&mut self, input: &[f64]) -> Vec<Array2<f64>> {
        let input = ArrayBase::from_vec(input.to_vec()).into_shape((input.len(), 1)).unwrap();
        self.predict_forword_propagation(vec![input]);

        self.network_layers
            .last()
            .unwrap()
            .clone()
            .outputs
            .unwrap()
    }
}
