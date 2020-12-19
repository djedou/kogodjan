use crate::{
    neural_traits::{LayerT, NetworkT},
    loss_functions::{LossFunction, GradFunction},
    maths::types::MatrixD,
    utils::{data_iter},
    optimizers::Optimizer
};




/// build a neural network with one layer and one neuron
/// # Example  
///
/// ```
/// use kongodjan::{
///    maths::types::MatrixD,
///    utils::{synthetic_data},
///    layers::single_layer::LrLayer,
///    activators::linear::purelin,
///    network_arch::LinearNetwork,
///    loss_functions::{squared_loss, squared_loss_gradient},
///    neural_traits::NetworkT,
///    optimizers::sgd
/// };
///
/// // rows are neurons and columns are inputs
/// let true_w = MatrixD::<f32>::from_row_slice(1, 2, &[
///    2.0, -3.4
/// ]);
///
/// let true_b = 4.2;
///
/// // the network is one layer with one neuron which receives two inputs
/// // then features are two rows(two inputs) and more columns
/// // labels are one row and more columns
/// let (features, labels) = synthetic_data(&true_w, true_b, 1000);
///
/// // build layer
/// let size_of_input_vec = 2; 
/// let layer = LrLayer::new(size_of_input_vec, purelin, None, 1);
///
/// // build the network
/// let mut network = LinearNetwork::new(features, layer, labels);
///
/// // train the network
/// network.train(0.01, Some(5), (squared_loss, squared_loss_gradient,sgd), 200);
///
/// //let pred = network.predict(&test_x);
/// //println!("output: {:?}", pred);
/// ```

#[derive(Debug, Clone)]
pub struct LinearNetwork<L>
where L: LayerT + Clone
{
    network_inputs: MatrixD<f32>,
    network_layers: L,
    network_outputs: MatrixD<f32>
}

impl<L> LinearNetwork<L>
where L: LayerT + Clone
{
    pub fn new( network_inputs: MatrixD<f32>, network_layers: L, network_outputs: MatrixD<f32>) -> LinearNetwork<L> {
        
        LinearNetwork {
            network_inputs,
            network_layers,
            network_outputs
        }
    }
}

impl<L> NetworkT for LinearNetwork<L> 
where L: LayerT + Clone
{
    fn train(&mut self, lr: f32, batch_size: Option<usize>, optimizers: (LossFunction, GradFunction, Optimizer), epoch: i32) {
        let loss_f: fn(output: &MatrixD<f32>, target: &MatrixD<f32>) -> MatrixD<f32> = optimizers.0;
        let grad_f: fn(errors: MatrixD<f32>) -> MatrixD<f32> = optimizers.1;
        let bat_size = match batch_size {
            Some(size) => size,
            None => 1
        };
        
        for round in 1..epoch {
            for (feature, label) in data_iter(bat_size, &self.network_inputs, &self.network_outputs){
                let forword_output = self.network_layers.forward(&feature);
                // calculate gradient here
                let gradient = grad_f(loss_f(&forword_output, &label));
                let _output = self.network_layers.backward(&lr, &bat_size, &gradient, &optimizers.2 );
            }
            
            let forword_output = self.network_layers.forward(&self.network_inputs);
            // calculate gradient here
            let error = loss_f(&forword_output, &self.network_outputs);
            println!("epoch: {:?} => loss: {:?}", round, error.mean());
            let gradient = grad_f(error);
            let _output = self.network_layers.backward(&lr, &self.network_inputs.nrows(), &gradient, &optimizers.2 );
        }
    }

    fn predict(&mut self, input: &MatrixD<f32>) -> MatrixD<f32> {
        self.network_layers.forward(&input)
    }
}