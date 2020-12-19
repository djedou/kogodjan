use ndarray::{ArrayD, Array, IxDyn};
use crate::activations::{FLOAT_SIZE_100};

/// each neuron in the output layer learns its own error
pub fn single_neuron_error_func(target: &ArrayD<f32>, output: ArrayD<f32>) -> (bool, ArrayD<f32>) {

    match output.shape() == target.shape() {
        true => {
            let size = output.shape()[0];
            let out_vec = output.as_slice().unwrap();
            let targ_vec = target.as_slice().unwrap();
            
            let mut error_vec = Vec::new();

            let mut should_learn = Vec::new();

            for n in 0..size {
                let f_o_val = out_vec[n];
                let f_t_val = targ_vec[n];
                
                let o_val = (f_o_val * FLOAT_SIZE_100).trunc() / FLOAT_SIZE_100;
                let t_val = (f_t_val * FLOAT_SIZE_100).trunc() / FLOAT_SIZE_100;

                if t_val == o_val {
                    should_learn.push(false);
                    println!("error1: {:?}", 0.0);
                    error_vec.push(0.0);
                } else {
                    //let error = ((0.5 * (t_val - o_val).exp2()) * FLOAT_SIZE_1000).trunc() / FLOAT_SIZE_1000;
                    if t_val > o_val {
                        let error = ((t_val - o_val) * FLOAT_SIZE_100).trunc() / FLOAT_SIZE_100;
                        println!("error2: {:?}", error);
                        should_learn.push(true);
                        error_vec.push(error);
                    }
                    else {
                        let error = ((o_val - t_val) * FLOAT_SIZE_100).trunc() / FLOAT_SIZE_100;
                        println!("error3: {:?}", error);
                        should_learn.push(true);
                        error_vec.push(error);

                    }
                } 
                
            }
            
            let error_deriv_outputs = Array::from_shape_fn(IxDyn(&[size, 1]), |args| {
                
                let arg = args[0];
                error_vec[arg]
                
            });

            println!("should lerarn: {:?}", should_learn);

            let learn = should_learn.contains(&true);
            
            (learn , error_deriv_outputs)
        },
        false => {
            panic!("the target and output should have same shape")
        }
    }
}

/*
let should_learn = |output: ArrayD<f32>, target: &ArrayD<f32>, total_error_target: f32| -> (bool, ArrayD<f32>) {

            match output.shape() == target.shape() {
                true => {
                    let size = output.shape()[0];
                    let out_vec = output.as_slice().unwrap();
                    let targ_vec = target.as_slice().unwrap();

                    let mut total_error = 0.0;

                    let mut error_deriv = Vec::new();

                    for n in 0..size {
                        let o_val = out_vec[n];
                        let t_val = targ_vec[n];
                        total_error = total_error + squared_error(&t_val, &o_val);
                        
                        //let er_de = -2.0 * (t_val - o_val);
                        let er_de = -(t_val - o_val);
                        error_deriv.push(er_de);
                        
                    }
                    //println!("error {}", total_error);

                    let error_deriv_outputs = Array::from_shape_fn(IxDyn(&[size, 1]), |args| {
            
                        let arg = args[0];
                        error_deriv[arg]

                    });

                    
                    (!(0.0 <= total_error && total_error <= total_error_target), error_deriv_outputs)
                },
                false => {
                    panic!("the target and output should have same shape")
                }
            }
        };
*/