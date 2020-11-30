use crate::neural_traits::LayerT;
use crate::neural_traits::NetworkT;
use ndarray::ArrayD;


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
    network_outputs: Vec<ArrayD<f64>>,
}

impl<L> Network<L>
where L: LayerT + Clone
{
    pub fn new(network_inputs: Vec<ArrayD<f64>>, network_layers: Vec<L>, network_outputs: Vec<ArrayD<f64>>) -> Network<L> {
        Network {
            network_inputs,
            network_layers,
            network_outputs
        }
    }
}

impl<L> NetworkT for Network<L> 
where L: LayerT + Clone
{
    fn train(&mut self) {

        let should_learn = |output: ArrayD<f64>, target: ArrayD<f64>| -> bool {

            match output.shape() == target.shape() {
                true => {
                    !(output == target)
                },
                false => {
                    panic!("the target and output should have same shape")
                }
            }
        };

        let inputs_size = self.network_inputs.len();

        let layers_size = self.network_layers.len();

        for ind in 0..inputs_size {
            let input = self.network_inputs[ind].clone();
            let output = self.network_outputs[ind].clone();

            'one_tour_loop: loop {
                // forward is implemented here
                let mut inp = input.clone();
                // go through all layers for one network input by a time
                for index in 0..layers_size {
                    let mut layer = self.network_layers[index].clone();
                    inp = layer.forward(inp);
                            
                }
    
                println!("output: {:?}", inp.clone());
                println!(" ");
                let should_learn = should_learn(inp, output.clone()); 
                match should_learn {
                    true => {
                        // backward is implemented here base on "inp"
                        break;
                    },
                    false => {
                        break 'one_tour_loop;
                    }
                }
            }

        }

        
    }
}

