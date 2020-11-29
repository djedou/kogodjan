use crate::neural_traits::LayerT;

/// build a neural network 
pub struct Network<L>
where L: LayerT
{
    layers: Vec<L>
}

impl<L> Network<L>
where L: LayerT
{
    pub fn new(layers: Vec<L>) -> Network<L> {
        Network {
            layers
        }
    }
}