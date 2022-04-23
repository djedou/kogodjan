use algo_diff::{
    maths::{Matrix, Axis, Array},
    graphs::{Graph, LinearGraph, SigmoidGraph, TanhGraph}
};
use crate::{
    activator::Activator
};
use serde_derive::{Serialize, Deserialize};
use rand::random;



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FcLayerIO {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activator: Activator
}


#[derive(Debug)]
pub struct FcLayer {
    weights: Matrix,
    biases: Matrix,
    //weights_m: Option<Matrix>,
    //biases_m: Option<Matrix>,
    activator_enum: Activator,
    activator: Box<dyn Graph<Input = Matrix, Gradient = Option<Matrix>, Output = Matrix>>,
    linear: Box<dyn Graph<Input = [Matrix; 3], Gradient = Option<[Matrix; 3]>, Output = Matrix>>,
    gradients: [Matrix; 2] // [weights, biases]
}


impl FcLayer {
    /// create a new Full Connected Layer  
    /// n_inputs(neuron input) is the number of input for the single neuron in the layer 
    /// n_neurons is the number of neurons for the layer    
    pub fn new(n_inputs: usize, n_neurons: usize, activator: Activator) -> FcLayer {
        // columns are for neurons and rows are for inputs
        //let max = 1.0 / (n_neurons as f64).sqrt();
        //let min = -max;

        let act: Box<dyn Graph<Input = Matrix, Gradient = Option<Matrix>, Output = Matrix>> = match activator {
            Activator::Sigmoid => Box::new(SigmoidGraph::new()),
            Activator::Tanh => Box::new(TanhGraph::new())
        };

        let weights = Matrix::from_shape_fn((n_inputs, n_neurons), |_| 2f64 * random::<f64>() - 1f64);
        let biases = Matrix::from_shape_fn((n_neurons, 1), |_| 2f64 * random::<f64>() - 1f64);

        FcLayer {
            weights, //: uniform(n_inputs, n_neurons, min, max, rng),
            biases, //: uniform(n_neurons, 1 , min, max, rng),
            //weights_m: None,
            //biases_m: None,
            activator: act,
            activator_enum: activator,
            linear: Box::new(LinearGraph::new()),
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }

    pub fn forward(&mut self, inputs: Matrix) -> Matrix {
        // adapt the size of the biases 
        let input_ncols = inputs.ncols();
        let mut new_biases = Matrix::zeros((self.biases.nrows(), 0));
        for _ in 0..input_ncols {
            new_biases.push_column(self.biases.column(0)).unwrap();
        }
        let linear_output = self.linear.forward([self.weights.clone().reversed_axes(), inputs, new_biases]);
        self.activator.forward(linear_output)
    }
    
    pub fn backward(&mut self, gradient: Option<Matrix>) -> Option<Matrix> {
        let activator_deriv = self.activator.backward(gradient);
        let linear_deriv = self.linear.backward(activator_deriv).unwrap();
        self.gradients = [linear_deriv[0].clone(), linear_deriv[2].clone()];
        Some(linear_deriv[1].clone())
    }

    pub fn update_parameters(&mut self, lr: f64, _momemtum: f64) {
        // update weights
        /*
        let weights_m = if let Some(w_m) = self.weights_m.clone() {
                            (momemtum * w_m) - (lr * self.gradients[0].clone().reversed_axes())
                        }
                        else {
                            - (lr * self.gradients[0].clone().reversed_axes())
                        };
        self.weights = self.weights.clone() + weights_m.clone(); // - (lr * self.gradients[0].clone().reversed_axes());
        self.weights_m = Some(weights_m);
        */
        self.weights = self.weights.clone() - (lr * self.gradients[0].clone().reversed_axes());

        // update biases
        /*
        let biases_deriv = self.gradients[1].sum_axis(Axis(1)).into_shape((self.biases.nrows(), 1)).unwrap();
        let biases_m = if let Some(b_m) = self.biases_m.clone() {
                            (momemtum * b_m) - (lr * biases_deriv)
                        }
                        else {
                            - (lr * biases_deriv)
                        };
        self.biases = self.biases.clone() + biases_m.clone();
        self.biases_m = Some(biases_m);
        */
        let biases_deriv = self.gradients[1].mean_axis(Axis(1)).unwrap().into_shape((self.biases.nrows(), 1)).unwrap();
        self.biases = self.biases.clone() - biases_deriv.clone();
    }

    pub fn save(&self) -> FcLayerIO {
        let mut weights: Vec<Vec<f64>> = vec![];
        let biases: Vec<f64> = self.biases.column(0).to_vec();

        for col in 0..self.weights.ncols() {
            weights.push(self.weights.column(col).to_vec());
        }

        FcLayerIO {
            weights,
            biases,
            activator: self.activator_enum.clone()
        }
    }
}


impl FcLayerIO {

    pub fn into(&self) -> FcLayer {
        let mut weights = Matrix::zeros((self.weights[0].len(), 0));
        let biases = Matrix::from_shape_vec((self.biases.len(), 1), self.biases.clone()).unwrap();
        
        for col in &self.weights {
            weights.push_column(Array::from_vec(col.clone()).view()).unwrap();
        }

        let act: Box<dyn Graph<Input = Matrix, Gradient = Option<Matrix>, Output = Matrix>> = match self.activator {
            Activator::Sigmoid => Box::new(SigmoidGraph::new()),
            Activator::Tanh => Box::new(TanhGraph::new())
        };

        FcLayer {
            weights,
            biases,
            //weights_m: None,
            //biases_m: None,
            activator: act,
            activator_enum: self.activator.clone(),
            linear: Box::new(LinearGraph::new()),
            gradients: [Matrix::zeros((1,1)), Matrix::zeros((1,1))]
        }
    }
}


#[cfg(test)]
mod fc_layer_test {
    use super::*;


    #[test]
    fn train() {
        let mut fc_layer = FcLayer::new(5, 7, Activator::Sigmoid);
        let inputs = Matrix::from_shape_vec((5, 3), vec![1.0,2.,0.,1.,0.,1.,0.,0.,1., 0.,1.0,2.,0.,1.,0.]).unwrap();
        let inputs_deriv = Matrix::from_shape_vec((7, 3), vec![1.0,2.,0.,1.,0.,1.,0.,0.,1., 0.,1.0,2.,0.,1.,0., 0.,1.,0.,1.,0.,0.]).unwrap();

        let layer_output = fc_layer.forward(inputs);
        println!("layer_output : {:#?}", layer_output);
        let input_grad = fc_layer.backward(Some(inputs_deriv)).unwrap();
        println!("input_grad : {:#?}", input_grad);
        fc_layer.update_parameters(0.001, 0.8);
    }
}