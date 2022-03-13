use ndarray::Array2;

pub trait NetworkT {
    fn forword_propagation(&mut self, features: &Array2<f64>);
    fn backword_propagation(&mut self, gradient: &Array2<f64>);
    fn update_parameters(&mut self, lr: f64);
    fn predict_forword_propagation(&mut self, features: &Array2<f64>);
    fn train(&mut self, lr: f64, batch_size: usize, epoch: i32);
    fn predict(&mut self, input: &[f64]) -> Array2<f64>;
    fn network_outpout(&self) -> Array2<f64>;
}


pub trait LayerT {
    fn forword(&mut self, input: &Array2<f64>) -> Array2<f64>;
    fn backword(&mut self, gradient: &Array2<f64>) -> Array2<f64>;
    fn predict_forword(&mut self, inputs: &Array2<f64>) -> Array2<f64>;
    fn update_parameters(&mut self, lr: f64);
}