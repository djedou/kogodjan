use crate::maths::types::MatrixD;
use rand::{thread_rng, Rng};
use crate::neural_traits::LayerT;
use crate::activators::types::{Activator, ActivatorDeriv};
use crate::optimizers::Optimizer;
use nalgebra::DVector;
//use crate::errors::Gradient;


/// Linear Regression Layer
#[derive(Debug, Clone)]
pub struct LrLayer {
    layer_id: i32,
    inputs: Option<MatrixD<f32>>,
    net_inputs: Option<MatrixD<f32>>,
    weights: MatrixD<f32>,
    biases: MatrixD<f32>,
    activator: Activator, 
    activator_deriv: Option<ActivatorDeriv>
}

impl LrLayer {
    /// create a new Linear Regression Layer  
    /// n_inputs(neuron input) is the number of input for the single neuron in the layer     
    pub fn new(n_inputs: usize, activator: Activator, activator_deriv: Option<ActivatorDeriv>, layer_id: i32) -> Self {
        let mut rng = thread_rng(); 

        // rows are for neurons and columns are for inputs per neuron
        let weights = MatrixD::<f32>::from_fn(1, n_inputs, |_a, _b| {
            
            let value = rng.gen::<f32>(); // generate float between 0.0 and 1.0
            value
        });

        let biases = MatrixD::<f32>::from_fn(1, 1 , |_a, _b| {
            
            let value = rng.gen::<f32>(); // generate float between 0.0 and 1.0
            value
        });

        LrLayer {
            layer_id,
            inputs: None,
            net_inputs: None,
            weights,
            biases,
            activator,
            activator_deriv

        }
    }
}



impl LayerT for LrLayer {

    fn forward(&mut self, inputs: &MatrixD<f32>) -> MatrixD<f32> {
        let activ_func: fn(MatrixD<f32>) -> MatrixD<f32> = self.activator;
        // calculate net_inputs
        let (wr, _wc) = self.weights.shape();
        let (_ir, ic) = inputs.shape();

        let mut net = MatrixD::<f32>::zeros(wr, ic);
        let mut net_input_part1 =  MatrixD::<f32>::zeros(wr, ic);
        self.weights.mul_to(&inputs, &mut net_input_part1);

        (0..ic).into_iter().for_each(|m| {
            let mut col_value = DVector::<f32>::from_element(wr, 0.0);
            
            net_input_part1.column(m).add_to(&self.biases, &mut col_value);
            net.set_column(m, &col_value);
            
        });
        self.inputs = Some(inputs.clone());
        self.net_inputs = Some(net.clone());
        activ_func(net)
        
    }

    fn backward(&mut self, lr: &f32, batch_size: &usize, gradient: &MatrixD<f32>, optimizer: &Optimizer) -> MatrixD<f32> {
        let grad = gradient.clone();
        let opt_func: fn(lr: &f32, batch_size: &usize, gradient: &MatrixD<f32>, param: &MatrixD<f32>, input: Option<&MatrixD<f32>>) -> MatrixD<f32> = *optimizer;

        let input = if let Some(ref inp) = self.inputs {
            inp
        } else {
            panic!("this layer does not have input");
        };

        self.weights = opt_func(&lr, &batch_size, &gradient, &self.weights, Some(&input));
        self.biases = opt_func(&lr, &batch_size, &gradient, &self.biases, None);

        grad
 
    }

    fn get_layer_id(&self) -> i32 {
        self.layer_id
    }

    fn set_weights(&mut self, weights: MatrixD<f32>) {
        self.weights = weights;
    }

    fn set_biases(&mut self, biases: MatrixD<f32>) {
        self.biases = biases;
    }

    fn get_weights(&self) -> Option<MatrixD<f32>> {None}

    fn get_biases(&self) -> Option<MatrixD<f32>> {None}
}
