
use crate::{
    layers::{FcLayer, LossLayer, FcLayerIO},
    activator::Activator,
    loss::Loss,
    networks::NetworkT,
    utils::data_iter,
    io::IO
};
use algo_diff::maths::Matrix;
use serde_derive::{Serialize, Deserialize};
use jfs::Store;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedForwardIO {
    network_layers: Vec<FcLayerIO>,
    loss: Loss
}


#[derive(Debug)]
pub struct FeedForward {
    network_layers: Vec<FcLayer>,
    loss_enum: Loss,
    loss: LossLayer,
}


impl FeedForward {
    pub fn new(layers: &[(usize, usize, Activator)], loss: Loss) -> FeedForward {
        
        let los = LossLayer::new(loss.clone());
        let mut fcn = FeedForward {
            network_layers: vec![],
            loss: los,
            loss_enum: loss
        };
        
        for i in 0..layers.len() {
            let (n_inputs, n_neurons, activator) = layers[i].clone();
            fcn.network_layers.push(FcLayer::new(n_inputs, n_neurons, activator));
        }

        fcn
    }

    fn training_forward(&mut self, inputs: &Matrix, targets: Matrix, lr: f64, momemtum: f64) -> f64 {

        // Forward
        let mut feat = inputs.clone();
        for j in 0..self.network_layers.len(){
            feat = self.network_layers[j].forward(feat);
        }
        
        // Loss 
        let loss = self.loss.forward(targets, feat);
        let loss_grad = self.loss.backward().unwrap();
        
        // Backward
        self.network_layers.reverse();
        let mut grad = Some(loss_grad);
        for j in 0..self.network_layers.len(){
            grad = self.network_layers[j].backward(grad);
        }
        self.network_layers.reverse();
        
        // Update
        for j in 0..self.network_layers.len(){
            self.network_layers[j].update_parameters(lr, momemtum);
        }

        loss
    }
}


impl NetworkT for FeedForward {
    fn train(&mut self, network_inputs: &[Matrix], network_outputs: &[Matrix], lr: f64, momemtum: f64, batch_size: usize, epoch: i32) {
        for round in 0..epoch {
            let mut erros: Vec<f64> = vec![];
            for (feature, label) in data_iter(batch_size, &network_inputs, &network_outputs) {
                
                let loss = self.training_forward(&feature, label, lr, momemtum);
                erros.push(loss);
            }
            
            for (features, labels) in data_iter(network_inputs.len(), &network_inputs, &network_outputs) {
                let loss = self.training_forward(&features, labels, lr, momemtum);
                println!("Epoch: {:?} => Loss: {:?}", round + 1, loss);
            }
        }
    }

    fn predict(&mut self, input: &Matrix) -> Matrix {
        // Forward
        let mut feat = input.clone();
        for j in 0..self.network_layers.len(){
            feat = self.network_layers[j].forward(feat);
        }

        feat
    }
}



impl IO for FeedForward {
    fn save(&self, path: &str) {
        let mut cfg = jfs::Config::default();
        cfg.pretty = true;
        cfg.indent = 4;
        let db = Store::new_with_cfg(path, cfg).unwrap();

        let mut network_layers: Vec<FcLayerIO> = vec![];
        for layer in &self.network_layers {
            network_layers.push(layer.save());
        }

        let data = FeedForwardIO {
            network_layers,
            loss: self.loss_enum.clone()
        };

        let _ = db.save(&data);
    }

    fn load(id: &str, data: &str) -> Self {
        let db = Store::new(data).unwrap();
        let data = db.get::<FeedForwardIO>(&id).unwrap();
        let network_layers = data.network_layers.iter().map(|n|  n.into()).collect();
        let los = LossLayer::new(data.loss.clone());
        FeedForward {
            network_layers,
            loss: los,
            loss_enum: data.loss
        }
    }
}
