use crate::{
    neural_traits::{LayerT, NetworkT},
    loss_functions::{LossFunction, GradFunction},
    maths::types::MatrixD,
    utils::{data_iter, Parameters},
    optimizers::Optimizer
};
use serde_json::{to_writer, from_reader};
use std::{io::BufReader, fs::File};



/// build a neural network with more layers, more neurons and fully connected  
/// # Example
/// ```
/// use kongodjan::{
///     maths::types::MatrixD,
///     utils::{synthetic_data_mat},
///     layers::multi_layers::FcLayer,
///     activators::non_linear::{logsig, logsig_deriv},
///     network_arch::PerceptronNetwork,
///     loss_functions::{squared_loss, squared_loss_gradient},
///     neural_traits::NetworkT,
///     optimizers::sgd
/// };
///
/// // rows are neurons and columns are inputs
/// // 7 inputs for 3 neurons
/// let true_w = MatrixD::<f32>::from_row_slice(3, 7, &[
///     2.0, -3.4, 2.0, -3.4, 2.0, -3.4, 1.0,
///     2.0, -3.4, 2.0, -3.4, 2.0, -3.4, 1.0,
///     2.0, -3.4, 2.0, -3.4, 2.0, -3.4, 1.0
/// ]);
///
/// let true_b = MatrixD::<f32>::from_row_slice(3, 1, &[
///     2.0,
///     -3.4,
///     1.0
/// ]);
///    
/// // the network is one layer with one neuron which receives two inputs
/// // then features are two rows(two inputs) and more columns
/// // labels one row and more columns
/// let (features, labels) = synthetic_data_mat(&true_w, true_b, 1000);
///    
/// // build layer
/// let (l1_n_neurons, l1_n_inputs) = (4, 7); 
/// let (l2_n_neurons, l2_n_inputs) = (3, 4);
/// 
/// let layer1 = FcLayer::new(l1_n_neurons, l1_n_inputs, logsig, Some(logsig_deriv), 1);
/// let layer2 = FcLayer::new(l2_n_neurons, l2_n_inputs, logsig, Some(logsig_deriv), 2);
///
/// let layers = vec![layer1, layer2];
///
/// // build the network
/// let mut network = PerceptronNetwork::new(features, layers, labels);
///
/// // train the networ
/// network.train(0.03, Some(10), (squared_loss, squared_loss_gradient,sgd), 20);
///
/// //let pred = network.predict(&test_x);
/// //println!("output: {:?}", pred);
/// ```
#[derive(Debug, Clone)]
pub struct PerceptronNetwork<L>
where L: LayerT + Clone
{
    network_db_id: String,
    network_inputs: MatrixD<f32>,
    network_layers: Vec<L>,
    network_outputs: MatrixD<f32>
}

impl<L> PerceptronNetwork<L>
where L: LayerT + Clone
{
    pub fn new(network_inputs: MatrixD<f32>, network_layers: Vec<L>, network_outputs: MatrixD<f32>) -> PerceptronNetwork<L> {
        
        PerceptronNetwork {
            network_db_id: String::new(),
            network_inputs,
            network_layers,
            network_outputs
        }
    }

    pub fn save_parameters(&mut self) {
        let mut paras: Vec<Parameters> = Vec::new();
        for l in &self.network_layers {
            if let Some(p) = l.save() {
                paras.push(p);
            }
        }

        to_writer(&File::create("db_params.json").unwrap(), &paras).unwrap();
        
    }

    pub fn restore_parameters(&mut self) {  
        if let Ok(file ) = File::open("db_params.json") {
            let reader = BufReader::new(file);
            
            if let Ok(data) = from_reader::<BufReader<File>, Vec<Parameters>>(reader) {
                for index in 0..data.len() {
                    if let Some(layer ) = data.iter().find(|d| d.layer_id == (index as i32)) {
                        self.network_layers.iter_mut().for_each(|l| {
                            if l.get_layer_id() == layer.layer_id {
                                l.set_weights(layer.layer_weights.clone());
                                l.set_biases(layer.layer_biases.clone());
                            }
                        });
                    }
                }
            }

        }
    }

    pub fn show_parameters(&self) {
        for layer in &self.network_layers {
            println!("layer_id: {} weights: {:#?}", layer.get_layer_id(), layer.get_weights());
            println!("layer_id: {} biases: {:#?}", layer.get_layer_id(), layer.get_biases());
        }
    }
}

impl<L> NetworkT for PerceptronNetwork<L> 
where L: LayerT + Clone
{
    fn train(&mut self, lr: f32, batch_size: Option<usize>, optimizers: (LossFunction, GradFunction, Optimizer), epoch: i32) {
        let loss_f: fn(output: &MatrixD<f32>, target: &MatrixD<f32>) -> MatrixD<f32> = optimizers.0;
        let loss_grad_f: fn(errors: MatrixD<f32>) -> MatrixD<f32> = optimizers.1;

        // get batch size
        let bat_size = match batch_size {
            Some(size) => size,
            None => 1
        };

        for round in 1..epoch {

            for (feature, label) in data_iter(bat_size, &self.network_inputs, &self.network_outputs){
                
                let mut input = feature.clone();
                
                // forword into all layers
                for index in 0..self.network_layers.len() {
                    let layer = &mut self.network_layers[index];
                    input = layer.forward(&input);   
                }
                
                // calculate the last layer gradient here
                let loss_grad = loss_grad_f(loss_f(&input, &label));
                let mut extern_gradient = loss_grad.clone();
                // backword into all layers for the last to the first
                let mut indexes: Vec<_> = (0..self.network_layers.len()).into_iter().collect();
                indexes.reverse();
                let len = indexes.len() - 1;
                for ind in 0..indexes.len() {
                    let i = len - ind;
                    let layer = &mut self.network_layers[i];
                    
                    extern_gradient = layer.backward(&lr, &bat_size, &extern_gradient, &optimizers.2);
                }
                //break;
            }
            
            let mut input = self.network_inputs.clone();
                
            // forword into all layers
            for index in 0..self.network_layers.len() {
                let layer = &mut self.network_layers[index];
                input = layer.forward(&input);   
            }
            
            // calculate the last layer gradient here
            let error = loss_f(&input, &self.network_outputs);
            println!("epoch: {:?} => loss: {:?}", round, error.mean());
            let loss_grad = loss_grad_f(error);
            let mut extern_gradient = loss_grad.clone();
            // backword into all layers for the last to the first
            let mut indexes: Vec<_> = (0..self.network_layers.len()).into_iter().collect();
            indexes.reverse();
            let len = indexes.len() - 1;
            for ind in 0..indexes.len() {
                let i = len - ind;
                let layer = &mut self.network_layers[i];
                
                extern_gradient = layer.backward(&lr, &bat_size, &extern_gradient, &optimizers.2);
            }
        }
    }

    fn predict(&mut self, input: &MatrixD<f32>) -> MatrixD<f32> {
        let mut inputs = input.clone();
                
        // forword into all layers
        for index in 0..self.network_layers.len() {
            let layer = &mut self.network_layers[index];
            inputs = layer.forward(&inputs);   
        }
        inputs
    }
}