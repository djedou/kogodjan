use crate::{
    neural_traits::{LayerT, NetworkT},
    loss_functions::{LossFunction, GradFunction},
    maths::types::MatrixD,
    utils::{data_iter},
    optimizers::Optimizer
};





/// build a neural network with more layers and more neurons 
#[derive(Debug, Clone)]
pub struct PerceptronNetwork<L>
where L: LayerT + Clone
{
    network_inputs: MatrixD<f64>,
    network_layers: Vec<L>,
    network_outputs: MatrixD<f64>
}

impl<L> PerceptronNetwork<L>
where L: LayerT + Clone
{
    pub fn new(network_inputs: MatrixD<f64>, network_layers: Vec<L>, network_outputs: MatrixD<f64>) -> PerceptronNetwork<L> {
        
        PerceptronNetwork {
            network_inputs,
            network_layers,
            network_outputs
        }
    }
}

impl<L> NetworkT for PerceptronNetwork<L> 
where L: LayerT + Clone
{
    fn train(&mut self, _lr: f64, batch_size: Option<usize>, optimizers: (LossFunction, GradFunction, Optimizer), epoch: i32) {
        let loss_f: fn(output: &MatrixD<f64>, target: &MatrixD<f64>) -> MatrixD<f64> = optimizers.0;
        let grad_f: fn(errors: MatrixD<f64>) -> MatrixD<f64> = optimizers.1;

        let bat_size = match batch_size {
            Some(size) => size,
            None => 1
        };

        for _round in 1..epoch {

            for (feature, label) in data_iter(bat_size, &self.network_inputs, &self.network_outputs){
                
                let mut input = feature.clone();

                // forword into all layers
                for index in 0..self.network_layers.len() {
                    let layer = &mut self.network_layers[index];
                    input = layer.forward(&input);   
                }
                // calculate gradient here
                let gradient = grad_f(loss_f(&input, &label));
                println!("gradient: {:?}", gradient);
                println!("");
    
                //let _output = self.network_layers.backward(&lr, &bat_size, &gradient, &optimizers.2 );
                break;
            }
            
            //let forword_output = self.network_layers.forward(&self.network_inputs);
            // calculate gradient here
            //let error = loss_f(&forword_output, &self.network_outputs);
            //println!("epoch: {:?} => loss: {:?}", round, error.mean());
            //let gradient = grad_f(error);
            //println!("gradient: {:?}", gradient);
            //let _output = self.network_layers.backward(&lr, &self.network_inputs.nrows(), &gradient, &optimizers.2 );
            break;

        }
    }

    fn predict(&mut self, _input: &MatrixD<f64>) -> MatrixD<f64> {
        MatrixD::zeros(3,5)
    }
}