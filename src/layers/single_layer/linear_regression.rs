use crate::maths::types::MatrixD;
use rand::{thread_rng, Rng};
use crate::neural_traits::LayerT;
use crate::activators::{Activator, ActivatorDeriv};
use crate::optimizers::Optimizer;
use nalgebra::DVector;
//use crate::errors::Gradient;


/// Linear Regression Layer
#[derive(Debug, Clone)]
pub struct LrLayer {
    inputs: Option<MatrixD<f64>>,
    net_inputs: Option<MatrixD<f64>>,
    weights: MatrixD<f64>,
    biases: MatrixD<f64>,
    activator: Activator, 
    activator_deriv: Option<ActivatorDeriv>
}

impl LrLayer {
    /// create a new Linear Regression Layer  
    /// n_inputs(neuron input) is the number of input for the single neuron in the layer     
    pub fn new(n_inputs: usize, activator: Activator, activator_deriv: Option<ActivatorDeriv>) -> Self {
        let mut rng = thread_rng(); 

        // rows are for neurons and columns are for inputs per neuron
        let weights = MatrixD::<f64>::from_fn(1, n_inputs, |_a, _b| {
            
            let value = rng.gen::<f64>(); // generate float between 0.0 and 1.0
            value
        });

        let biases = MatrixD::<f64>::from_fn(1, 1 , |_a, _b| {
            
            let value = rng.gen::<f64>(); // generate float between 0.0 and 1.0
            value
        });

        LrLayer {
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

    fn forward(&mut self, inputs: &MatrixD<f64>) -> MatrixD<f64> {
        let activ_func: fn(MatrixD<f64>) -> MatrixD<f64> = self.activator;
        // calculate net_inputs
        let (wr, _wc) = self.weights.shape();
        let (_ir, ic) = inputs.shape();

        let mut net = MatrixD::<f64>::zeros(wr, ic);
        let mut net_input_part1 =  MatrixD::<f64>::zeros(wr, ic);
        self.weights.mul_to(&inputs, &mut net_input_part1);

        (0..ic).into_iter().for_each(|m| {
            let mut col_value = DVector::<f64>::from_element(wr, 0.0);
            
            net_input_part1.column(m).add_to(&self.biases, &mut col_value);
            net.set_column(m, &col_value);
            
        });
        self.net_inputs = Some(net.clone());
        activ_func(net)
        
    }

    fn backward(&mut self, lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, optimizer: &Optimizer) -> MatrixD<f64> {
        let grad = gradient.clone();
        let opt_func: fn(lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, param: &MatrixD<f64>) -> MatrixD<f64> = *optimizer;
        
        self.weights = opt_func(&lr, &batch_size, &gradient, &self.weights);
        self.biases = opt_func(&lr, &batch_size, &gradient, &self.biases);
        
        grad
 
    }

    
}
