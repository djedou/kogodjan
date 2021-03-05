use crate::maths::types::MatrixD;

#[derive(Serialize,Deserialize, Debug)]
pub struct Parameters {
    pub layer_id: i32,
    pub layer_weights: MatrixD<f64>,
    pub layer_biases: MatrixD<f64>
}