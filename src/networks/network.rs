use crate::neural_traits::LayerT;
use crate::neural_traits::NetworkT;
use ndarray::ArrayD;

/// build a neural network 
#[derive(Debug, Clone)]
pub struct Network<L>
where L: LayerT + Clone
{
    network_inputs: Vec<L::Inputs>,
    network_layers: Vec<L>,
    network_outputs: ArrayD<f64>,
}

impl<L> Network<L>
where L: LayerT + Clone
{
    pub fn new(network_inputs: Vec<L::Inputs>, network_layers: Vec<L>, network_outputs: ArrayD<f64>) -> Network<L> {
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

        let layers_size = self.network_layers.len();
        let inputs = self.network_inputs.clone();

        inputs.iter().for_each(|input| {

            let mut inp = input.clone();
            for index in 0..layers_size {

                let mut layer = self.network_layers[index].clone();
    
                inp = layer.forward(inp);

                println!("outputs: {:#?}", inp);
                
            }

        });

        
    }
}