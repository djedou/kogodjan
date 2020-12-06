use crate::neural_traits::LayerT;
use crate::neural_traits::NetworkT;
use ndarray::{ArrayD};
use crate::errors::{last_layer_gradient, others_layers_gradient};
use crate::errors::NetworkErrorFunction;


/// build a neural network 
///
/// # Example  
///
/// ```
/// use kongodjan::architectures::perceptron::FullConnectedLayer;
/// use kongodjan::ndarray::{Array, IxDyn};
/// use kongodjan::networks::Network;
/// use kongodjan::neural_traits::NetworkT;
/// use kongodjan::activations::tansig;
///
/// let layer1 = FullConnectedLayer::new(2, 3, tansig);
/// let layer2 = FullConnectedLayer::new(3, 2, tansig);
/// 
/// let layers = vec![layer1, layer2];
///
/// let mut inputs = Vec::new();
/// let mut outputs = Vec::new();
///
/// // each input has one output
/// for n in 1..3 {
///     let input = Array::from_shape_fn(IxDyn(&[2, 1]), |args| {
///        ((1 + args[0]) * n) as f64
///     });
///
///     inputs.push(input);
///
///     let output = Array::from_shape_fn(IxDyn(&[2, 1]), |args| {
///        ((1 + args[0]) * 2) as f64
///     });
///
///    outputs.push(output);
/// }
///
/// let mut network = Network::new(inputs, layers, outputs);
///
///  network.train();
/// ```
///

#[derive(Debug, Clone)]
pub struct Network<L>
where L: LayerT + Clone
{
    network_inputs: Vec<ArrayD<f64>>,
    network_layers: Vec<L>,
    network_targets: Vec<ArrayD<f64>>,
}

impl<L> Network<L>
where L: LayerT + Clone
{
    pub fn new(network_inputs: Vec<ArrayD<f64>>, network_layers: Vec<L>, network_targets: Vec<ArrayD<f64>>) -> Network<L> {
        Network {
            network_inputs,
            network_layers,
            network_targets
        }
    }
}

impl<L> NetworkT for Network<L> 
where L: LayerT + Clone
{
    fn train(&mut self, rate: f64, error_func: NetworkErrorFunction) {

        let error: fn(target: &ArrayD<f64>, output: ArrayD<f64>) -> (bool, ArrayD<f64>) = error_func;

        let inputs_size = self.network_inputs.len();

        let layers_size = self.network_layers.len();

        for ind in 0..inputs_size {
            //println!("tours: {}", ind);
            let input = &self.network_inputs[ind];
            let target = &self.network_targets[ind];

            'one_tour_loop: loop {
                // forward is implemented here
                let mut inp = input.clone();
                //println!("inputs: {:?}", inp);
                //println!("targets: {:?}", target);
                // go through all layers for one network input by a time
                for index in 0..layers_size {
                    let layer = &mut self.network_layers[index];
                    inp = layer.forward(inp);   
                }
                println!("network output: {:?}", inp);
                println!("target: {:?}", target);
                let (should_learn, error_deriv )= error(target,inp); 
                println!("network error deriv: {:?}", error_deriv);
                match should_learn {
                    true => {
                        // backward is implemented here base on "inp"
                        let mut error_der_inp = error_deriv.clone();
                        
                        for i in 0..layers_size {
                            let index = (layers_size -1) - i;
                            if index == layers_size - 1 {
                                let layer = &mut self.network_layers[index];
                                error_der_inp = layer.backward(error_der_inp, rate, last_layer_gradient);
                            } else {
                                let layer = &mut self.network_layers[index];
                                error_der_inp = layer.backward(error_der_inp, rate, others_layers_gradient);
                            }
                        }
                        //break;
                    },
                    false => {
                        println!("djed");
                        break 'one_tour_loop;
                    }
                }
            }

        }

        
    }
}