mod feed_forward;


pub use feed_forward::*;
use algo_diff::maths::Matrix;


pub trait NetworkT {
    fn train(&mut self, network_inputs: &Matrix, network_outputs: &Matrix, lr: f64, batch_size: usize, epoch: i32);
    fn predict(&mut self, input: &Matrix) -> Matrix;
}
