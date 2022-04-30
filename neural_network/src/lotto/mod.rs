mod layers;

use algo_diff::{
    graphs::{CrossEntropyGraph, CopyGraph, Graph},
    maths::Matrix,
};
use serde_derive::{Serialize, Deserialize};
pub(crate) use layers::*;
use jfs::Store;
use crate::{
    lotto::{SoftmaxLayer, SoftmaxLayerIO},
    utils::data_iter_v2,
    io::IO
};


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LottoNetworkIO {
    hidden: HiddenLayerIO,
    softmax: Vec<SoftmaxLayerIO>,
}


#[derive(Debug)]
pub struct LottoNetwork {
    hidden: HiddenLayer,
    copy: CopyGraph,
    softmax: Vec<SoftmaxLayer>,
    losses: Vec<CrossEntropyGraph>
}

impl LottoNetwork {
    pub fn new(h_neurons: usize, h_inputs: usize, s_neurons: usize, s_inputs: usize) -> LottoNetwork {

        let hidden = HiddenLayer::new(h_neurons, h_inputs);
        let mut softmax = vec![];
        let mut losses = vec![];
        for _ in 0..10 {
            softmax.push(SoftmaxLayer::new(s_neurons, s_inputs));
            losses.push(CrossEntropyGraph::new());
        }
        let copy = CopyGraph::new(10);

        LottoNetwork {
            hidden,
            copy,
            softmax,
            losses
        }

    }

    fn local_train(&mut self, feature: Matrix, label: &[Matrix], lr: f64) -> f64 {

        // Forward
        let hidden_output = self.hidden.forward(feature);
        let copy_output = self.copy.forward(hidden_output);
        let mut losses = vec![];
        for i in (0..copy_output.len()).into_iter() {
            let output = self.softmax[i].forward(copy_output[i].clone());
            let loss = self.losses[i].forward([label[i].clone(), output]);
            losses.push(loss);
        }
        
        // Backward
        let mut gradients = vec![];
        for i in (0..copy_output.len()).into_iter() {
            let loss_grad = self.losses[i].backward(None);
            let output_grad = self.softmax[i].backward(loss_grad);
            gradients.push(output_grad.unwrap());
        }
        let copy_grad = self.copy.backward_with_more_gradients(Some(&gradients));
        let _ = self.hidden.backward(copy_grad);

        // Update
        self.hidden.update_parameters(lr);
        for i in (0..copy_output.len()).into_iter() {
            self.softmax[i].update_parameters(lr);
        }

        let sum: f64 = losses.iter().sum();

        sum / losses.len() as f64
    }


    pub fn train(&mut self, network_inputs: &[Matrix], network_outputs: Vec<Vec<Matrix>>, lr: f64, batch_size: usize, epoch: i32) {
        for round in 0..epoch {
            let mut erros: Vec<f64> = vec![];
            for (feature, label) in data_iter_v2(batch_size, &network_inputs, &network_outputs) {
                
                let _ = self.local_train(feature, &label, lr);
            }
            for (features, labels) in data_iter_v2(network_inputs.len(), &network_inputs, &network_outputs) {
                let loss = self.local_train(features, &labels, lr);
                println!("Epoch: {:?} => Loss: {:?}", round + 1, loss);
            }
        }
    }

    pub fn predict(&mut self, input: Matrix) -> Vec<Matrix> {
        // Forward
        let hidden_output = self.hidden.forward(input);
        let copy_output = self.copy.forward(hidden_output);
        let mut outputs = vec![];
        for i in (0..copy_output.len()).into_iter() {
            let output = self.softmax[i].forward(copy_output[i].clone());
            outputs.push(output);
        }

        outputs
    }
}



impl IO for LottoNetwork {
    fn save(&self, path: &str) {
        let mut cfg = jfs::Config::default();
        cfg.pretty = true;
        cfg.indent = 4;
        let db = Store::new_with_cfg(path, cfg).unwrap();

        let mut softmax: Vec<SoftmaxLayerIO> = vec![];
        for layer in &self.softmax {
            softmax.push(layer.save());
        }

        let hidden = self.hidden.save();

        let data = LottoNetworkIO {
            softmax,
            hidden
        };

        let _ = db.save(&data);
    }

    fn load(id: &str, data: &str) -> Self {
        let db = Store::new(data).unwrap();
        let data = db.get::<LottoNetworkIO>(&id).unwrap();
        let softmax: Vec<SoftmaxLayer> = data.softmax.iter().map(|n|  n.into()).collect();
        let hidden = data.hidden.into_layer();
        
        let mut losses = vec![];
        for _ in 0..10 {
            losses.push(CrossEntropyGraph::new());
        }
        let copy = CopyGraph::new(10);
        LottoNetwork {
            softmax,
            hidden,
            losses,
            copy
        }
    }
}
