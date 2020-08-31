use nalgebra::{Matrix, VecStorage, Dynamic};

//use nalgebra::{ Dynamic};
//use typenum::{P3, Integer};


pub type PerceptronInput = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic,Dynamic>>;
pub type PerceptronOutput = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic,Dynamic>>;
pub type PerceptronWeights = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic,Dynamic>>;

pub struct Perceptron {
    pub input: PerceptronInput,
    pub output: PerceptronOutput,
    pub hiddens: Option<Vec<i32>>,// vec![2], len = hidden number, an index value = number of neurons in hidden
                                // so here, one hidden with two neurons
    pub weights: Option<Vec<PerceptronWeights>>,
}


impl Perceptron {
    fn initiate_weights(&mut self) {
        let hiddens = &self.hiddens;
        match hiddens {
            Some(hidden) => {
                let len = hidden.len();
                let mut w_vec: Vec::<PerceptronWeights> = Vec::new(); 
                let (nrows, _) = self.input.shape();
                let mut input_size = nrows;

                for i in 0..len {
                    let ncols = hidden[i];
                    let weights = PerceptronWeights::new_random(ncols as usize, input_size);
                    w_vec.push(weights);
                    input_size = ncols as usize;
                }

                let (output_rows, _) = self.output.shape();
                let output_weights = PerceptronWeights::new_random(output_rows as usize,input_size);
                w_vec.push(output_weights);
                
                self.weights = Some(w_vec);
            }
            None => {
                let (input_rows, _input_cols) = self.input.shape();
                let (output_rows, _outpu_cols) = self.output.shape();

                let weights = PerceptronWeights::new_random(output_rows,input_rows);
                let weights_vec = vec![weights];
                self.weights =  Some(weights_vec)
            }
        }
    }
    pub fn get_weights(&mut self) {
        let weights = self.weights.as_ref();
        
        match weights {
            Some(w) => {
                let weights_list = w.iter();
                for a in weights_list {
                    println!("weights {} ", a);
                } 
            }
            None => unimplemented!()
        } 
    }

    pub fn train(&mut self) {
        self.initiate_weights();
        
        println!("input: {} output: {} hidden: {:?}",self.input, self.output, self.hiddens);
        self.get_weights();

    }
}
