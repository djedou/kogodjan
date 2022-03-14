use ndarray::Array2;

pub trait NetworkT {
    fn forword_propagation(&mut self, features: Vec<Array2<f64>>);
    fn backword_propagation(&mut self, gradient: Vec<Array2<f64>>);
    fn update_parameters(&mut self, lr: f64);
    fn predict_forword_propagation(&mut self, features: Vec<Array2<f64>>);
    fn train(&mut self, lr: f64, batch_size: usize, epoch: i32);
    fn predict(&mut self, input: &[f64]) -> Vec<Array2<f64>>;
    fn network_outpout(&self) -> Vec<Array2<f64>>;
}

pub trait LayerT {
    fn forword(&mut self, input: Vec<Array2<f64>>) -> Vec<Array2<f64>>;
    fn backword(&mut self, gradient: Vec<Array2<f64>>) -> Vec<Array2<f64>>;
    fn predict_forword(&mut self, inputs: Vec<Array2<f64>>) -> Vec<Array2<f64>>;
    fn update_parameters(&mut self, lr: f64);
}