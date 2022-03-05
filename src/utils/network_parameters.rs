use crate::maths::Matrix;

//#[derive(Serialize, Deserialize, Debug)]
pub struct Parameters {
    pub layer_id: i32,
    pub layer_weights: Matrix<f64>,
    pub layer_biases: Matrix<f64>
}
