//maths::types::MatrixD,
/*, Parameters*/
use crate::{
    loss_functions::{LossFunction, GradFunction},
    utils::{data_iter/*, extern_gradient*/},
    neural_traits::{LayerT, NetworkT},
};
//use serde_json::{to_writer, from_reader};
//use std::{io::BufReader, fs::File};
use crate::maths::Matrix;


/*
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
/// let true_w = MatrixD::<f64>::from_row_slice(3, 7, &[
///     2.0, -3.4, 2.0, -3.4, 2.0, -3.4, 1.0,
///     2.0, -3.4, 2.0, -3.4, 2.0, -3.4, 1.0,
///     2.0, -3.4, 2.0, -3.4, 2.0, -3.4, 1.0
/// ]);
///
/// let true_b = MatrixD::<f64>::from_row_slice(3, 1, &[
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
*/


#[derive(Debug, Clone)]
pub struct FCNetwork<L>
where L: LayerT + Clone
{
    network_db_id: String,
    network_inputs: Matrix<f64>,
    network_layers: Vec<L>,
    network_outputs: Matrix<f64>
}

impl<L> FCNetwork<L>
where L: LayerT + Clone
{
    pub fn new(network_inputs: Matrix<f64>, network_layers: Vec<L>, network_outputs: Matrix<f64>) -> FCNetwork<L> {
        
        FCNetwork {
            network_db_id: String::new(),
            network_inputs,
            network_layers,
            network_outputs
        }
    }

    /*pub fn save_parameters(&mut self) {
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
                    if let Some(layer ) = data.iter().find(|d| d.layer_id == (index as i32) + 1) {
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
    }*/
}

impl<L> NetworkT for FCNetwork<L> 
where L: LayerT + Clone
{
    fn train(&mut self, lr: f64, batch_size: usize, optimizers: (LossFunction, GradFunction), epoch: i32) -> Result<(), String> {

        
        let loss_f: fn(output: &Matrix<f64>, target: &Matrix<f64>) -> f64 = optimizers.0;
        let loss_grad_f: fn(output: &Matrix<f64>, target: &Matrix<f64>) -> Matrix<f64> = optimizers.1;
        
        //let mut round: i64 = 0;
        for _round in 0..epoch {
            for (feature, label) in data_iter(batch_size, &self.network_inputs, &self.network_outputs) {
                let mut input = feature.clone();
                
                // forword into all layers
                for index in 0..self.network_layers.len() {
                    let layer = &mut self.network_layers[index];
                    input = layer.forward(&input)?;
                }
            
                let cost = loss_f(&input, &label);
                println!("error: {:#?}", cost);

                let mut gradient = loss_grad_f(&input, &label);
                
                // backword into all layers for the last to the first
                let indexes: Vec<_> = (0..self.network_layers.len()).into_iter().rev().collect();
                let lastlayer_index = indexes.len() - 1;

                for ind in indexes.iter() {
                    if ind == &lastlayer_index {
                        // prepare gradient here
                        let layer = &mut self.network_layers[lastlayer_index];
                        let net_input = layer.get_net_inputs().unwrap();
                        let der = layer.get_activator_deriv().unwrap();
                        let net_input_der = der(net_input);

                        let local_gradient = net_input_der.get_data() * gradient.get_data();
                        let local_gradient_mat = Matrix::new_from_array2(&local_gradient);

                        gradient = local_gradient_mat.clone();

                        layer.backward(lr, &local_gradient_mat);

                    }
                    else {
                        // prepare gradient here
                        let m_1_weights = &self.network_layers[*ind + 1]
                                .get_weights()
                                .unwrap()
                                .get_data().reversed_axes();

                        let layer = &mut self.network_layers[*ind];
                        let net_input = layer.get_net_inputs().unwrap();
                        let der = layer.get_activator_deriv().unwrap();
                        let net_input_der = der(net_input);
                        
                        let wp = m_1_weights.dot(&gradient.get_data());
                        let wp_grad = wp * net_input_der.get_data();
                        
                        let new_gradient = Matrix::new_from_array2(&wp_grad);

                        gradient = new_gradient.clone();
                        
                        layer.backward(lr, &new_gradient);
                    }

                }
            } 
            
        }
        Ok(())
    }

    fn predict(&mut self, input: &Matrix<f64>) -> Result<Matrix<f64>, String> {
        let mut inputs = input.clone();
                
        // forword into all layers
        for index in 0..self.network_layers.len() {
            let layer = &mut self.network_layers[index];
            inputs = layer.forward(&inputs)?;
        }

        Ok(inputs)
    }
}
