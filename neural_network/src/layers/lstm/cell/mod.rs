use crate::{
    layers::{FcLayer/*, LossLayer, FcLayerIO*/},
    activator::Activator,
    /*loss::Loss,
    networks::NetworkT,
    utils::data_iter,
    io::IO*/
};


#[derive(Debug)]
pub struct Cell {
    forget_gate: FcLayer,
    input_gate: FcLayer,
    candidate: FcLayer,
    output_gate: FcLayer
}


impl Cell {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Cell {

        Cell {
            forget_gate: FcLayer::new(n_inputs, n_neurons, Activator::Sigmoid),
            input_gate: FcLayer::new(n_inputs, n_neurons, Activator::Sigmoid),
            candidate: FcLayer::new(n_inputs, n_neurons, Activator::Tanh),
            output_gate: FcLayer::new(n_inputs, n_neurons, Activator::Sigmoid)
        }
    }
}