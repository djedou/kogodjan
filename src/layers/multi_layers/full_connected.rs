use crate::maths::types::MatrixD;
use rand::{thread_rng, Rng};
use crate::neural_traits::LayerT;
use crate::activators::types::{Activator, ActivatorDeriv};
use crate::optimizers::Optimizer;
use nalgebra::DVector;


/// Linear Regression Layer
#[derive(Debug, Clone)]
pub struct FcLayer {
    inputs: Option<MatrixD<f64>>,
    net_inputs: Option<MatrixD<f64>>,
    weights: MatrixD<f64>,
    biases: MatrixD<f64>,
    activator: Activator, 
    activator_deriv: Option<ActivatorDeriv>
}

impl FcLayer {
    /// create a new Linear Regression Layer  
    /// n_inputs(neuron input) is the number of input for the single neuron in the layer 
    /// n_neurons is the number of neurons for the layer    
    pub fn new(n_neurons: usize, n_inputs: usize, activator: Activator, activator_deriv: Option<ActivatorDeriv>) -> Self {
        let mut rng = thread_rng(); 

        // rows are for neurons and columns are for inputs per neuron
        let weights = MatrixD::<f64>::from_fn(n_neurons, n_inputs, |_a, _b| {
            
            let value = rng.gen::<f64>(); // generate float between 0.0 and 1.0
            value
        });

        let biases = MatrixD::<f64>::from_fn(n_neurons, 1 , |_a, _b| {
            
            let value = rng.gen::<f64>(); // generate float between 0.0 and 1.0
            value
        });

        FcLayer {
            inputs: None,
            net_inputs: None,
            weights,
            biases,
            activator,
            activator_deriv

        }
    }
}



impl LayerT for FcLayer {

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
        self.inputs = Some(inputs.clone());
        self.net_inputs = Some(net.clone());
        activ_func(net)
        
    }

    fn backward(&mut self, lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, optimizer: &Optimizer) -> MatrixD<f64> {

        let old_weights = self.weights.clone().transpose();
        // get the deriv function 
        let deriv: fn(MatrixD<f64>) -> MatrixD<f64> = if let Some(d) = self.activator_deriv {
            d
        } else {
            panic!("please provide derivative for all activators");
        };
         
        // first calculate the net_input gradient
        let mut layer_grad = if let Some(ref net) = self.net_inputs {
            deriv(net.clone())
        } else {
            panic!("this layer can not be used");
        };

        // calulate the layer gradient
        layer_grad.zip_apply(&gradient, |n,g| n * g);

        let opt_func: fn(lr: &f64, batch_size: &usize, gradient: &MatrixD<f64>, param: &MatrixD<f64>, input: Option<&MatrixD<f64>>) -> MatrixD<f64> = *optimizer;
        
        let input = if let Some(ref inp) = self.inputs {
            inp
        } else {
            panic!("this layer does not have input");
        };

        self.weights = opt_func(&lr, &batch_size, &layer_grad, &self.weights, Some(&input));
        self.biases = opt_func(&lr, &batch_size, &layer_grad, &self.biases, None);

        let mut return_grad = MatrixD::<f64>::zeros(old_weights.nrows(), layer_grad.ncols());
        old_weights.mul_to(&layer_grad, &mut return_grad);
        return_grad
 
    }

    
}
