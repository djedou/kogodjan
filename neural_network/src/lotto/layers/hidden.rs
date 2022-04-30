use algo_diff::{
    maths::{Matrix, Axis, Array},
    graphs::{SigmoidGraph, LinearGraph, Graph}
};
use serde_derive::{Serialize, Deserialize};
use rand::random;


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenLayerIO {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>
}



#[derive(Debug)]
pub struct HiddenLayer {
    weights: Matrix,
    biases: Matrix,
    activator: SigmoidGraph,
    linear: LinearGraph,
    gradients: [Matrix; 2] // [weights, biases]
}


impl HiddenLayer {
    pub fn new(n_neurons: usize, n_inputs: usize) -> HiddenLayer {
        let weights = Matrix::from_shape_fn((n_neurons, n_inputs), |_| 2f64 * random::<f64>() - 1f64);
        let biases = Matrix::from_shape_fn((n_neurons, 1), |_| 2f64 * random::<f64>() - 1f64);

        HiddenLayer {
            weights,
            biases,
            activator: SigmoidGraph::new(),
            linear: LinearGraph::new(),
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }

    pub fn forward(&mut self, inputs: Matrix) -> Matrix {
        let input_ncols = inputs.ncols();
        let mut new_biases = Matrix::zeros((self.biases.nrows(), 0));
        for _ in 0..input_ncols {
            new_biases.push_column(self.biases.column(0)).unwrap();
        }
        let linear_output = self.linear.forward([self.weights.clone(), inputs, new_biases]);
        self.activator.forward(linear_output)
    }

    pub fn backward(&mut self, gradient: Option<Matrix>) -> Option<Matrix> {
        let activator_deriv = self.activator.backward(gradient);
        let linear_deriv = self.linear.backward(activator_deriv).unwrap();
        self.gradients = [linear_deriv[0].clone(), linear_deriv[2].clone()];
        Some(linear_deriv[1].clone())
    }

    pub fn update_parameters(&mut self, lr: f64) {
        self.weights = self.weights.clone() - (lr * self.gradients[0].clone());

        let biases_deriv = self.gradients[1].mean_axis(Axis(1)).unwrap().into_shape((self.biases.nrows(), 1)).unwrap();
        self.biases = self.biases.clone() - biases_deriv.clone();
    }

    pub fn save(&self) -> HiddenLayerIO {
        let mut weights: Vec<Vec<f64>> = vec![];
        let biases: Vec<f64> = self.biases.column(0).to_vec();

        for col in 0..self.weights.ncols() {
            weights.push(self.weights.column(col).to_vec());
        }

        HiddenLayerIO {
            weights,
            biases,
        }
    }
}


impl HiddenLayerIO {
    pub fn into_layer(&self) -> HiddenLayer {
        let mut weights = Matrix::zeros((self.weights[0].len(), 0));
        let biases = Matrix::from_shape_vec((self.biases.len(), 1), self.biases.clone()).unwrap();
        
        for col in &self.weights {
            weights.push_column(Array::from_vec(col.clone()).view()).unwrap();
        }

        HiddenLayer {
            weights,
            biases,
            activator: SigmoidGraph::new(),
            linear: LinearGraph::new(),
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }
}