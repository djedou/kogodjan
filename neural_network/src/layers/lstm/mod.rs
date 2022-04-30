use algo_diff::{
    maths::{Matrix, Axis},
    graphs::{DotGraph, Graph, SigmoidGraph, TanhGraph, MulGraph, AddGraph}
};


#[derive(Debug)]
pub struct LstmLayer {
    gradients: Matrix
}


impl LstmLayer {
    pub fn new() -> LstmLayer {
    
        LstmLayer {
            gradients: Matrix::zeros((0,0))
        }
    }

    pub fn forward(&mut self, parameters: &Matrix, inputs: Matrix, hidden_state: Matrix, memory_state: Matrix) -> (Matrix, Matrix, Matrix) {
        let mut inputs_hidden_bias = Matrix::zeros((0, 1));
        inputs_hidden_bias.append(Axis(0), inputs.view()).unwrap();
        inputs_hidden_bias.append(Axis(0), hidden_state.view()).unwrap();
        inputs_hidden_bias.append(Axis(0), Matrix::ones((1,1)).view()).unwrap();

        // Step 1:
        let mut step_1 = DotGraph::new();
        let step_1_out = step_1.forward([parameters.clone(), inputs_hidden_bias]);
        
        // Step 1.a
        let step_1_out_vec: Vec<Vec<f64>> = step_1_out.column(0).to_vec().chunks(hidden_state.nrows()).into_iter().map(|a| a.to_owned()).collect();
        // forget_gate
        let forget_gate = Matrix::from_shape_vec((step_1_out_vec[0].len(), 1), step_1_out_vec[0].to_owned()).unwrap();
        // input_gate
        let input_gate = Matrix::from_shape_vec((step_1_out_vec[1].len(), 1), step_1_out_vec[1].to_owned()).unwrap();
        // output_gate
        let output_gate = Matrix::from_shape_vec((step_1_out_vec[2].len(), 1), step_1_out_vec[2].to_owned()).unwrap();
        // candidate
        let candidate = Matrix::from_shape_vec((step_1_out_vec[3].len(), 1), step_1_out_vec[3].to_owned()).unwrap();

        // Step 1.b:
        let mut forget_gate_sigmoid = SigmoidGraph::new();
        let forget_gate_sigmoid_out = forget_gate_sigmoid.forward(forget_gate);

        // Step 1.c:
        let mut input_gate_sigmoid = SigmoidGraph::new();
        let input_gate_sigmoid_out = input_gate_sigmoid.forward(input_gate);

        // Step 1.d:
        let mut output_gate_sigmoid = SigmoidGraph::new();
        let output_gate_sigmoid_out = output_gate_sigmoid.forward(output_gate);

        // Step 1.e:
        let mut candidate_tanh = TanhGraph::new();
        let candidate_sigmoid_out = candidate_tanh.forward(candidate);

        // Step 2:
        let mut forget_gate_memory_state_step = MulGraph::new();
        let forget_gate_memory_state = forget_gate_memory_state_step.forward([forget_gate_sigmoid_out, memory_state]);

        // Step 3:
        let mut input_gate_candidate_step = MulGraph::new();
        let input_gate_candidate = input_gate_candidate_step.forward([input_gate_sigmoid_out, candidate_sigmoid_out]);

        // Step 4:
        let mut forget_gate_memory_state_input_gate_candidate_step = AddGraph::new();
        let new_memory_state = forget_gate_memory_state_input_gate_candidate_step.forward([forget_gate_memory_state, input_gate_candidate]);

        // Step 5:
        let mut tanh_memory_state = TanhGraph::new();
        let tanh_memory_state_out = tanh_memory_state.forward(new_memory_state.clone());

        // Step 6:
        let mut new_hidden_state_step = MulGraph::new();
        let new_hidden_state = new_hidden_state_step.forward([output_gate_sigmoid_out, tanh_memory_state_out]);
        
        // ######## Gradients #######
        // Step 6 deriv:
        let new_hidden_state_step_deriv = new_hidden_state_step.backward(None).unwrap();

        // Step 5 deriv:
        let tanh_memory_state_deriv = tanh_memory_state.backward(Some(new_hidden_state_step_deriv[1].clone()));

        // Step 4 deriv:
        let forget_gate_memory_state_input_gate_candidate_step_deriv = forget_gate_memory_state_input_gate_candidate_step.backward(tanh_memory_state_deriv).unwrap();

        // Step 3 deriv:
        let input_gate_candidate_step_deriv = input_gate_candidate_step.backward(Some(forget_gate_memory_state_input_gate_candidate_step_deriv[1].clone())).unwrap();

        // Step 2 deriv:
        let forget_gate_memory_state_step_deriv = forget_gate_memory_state_step.backward(Some(forget_gate_memory_state_input_gate_candidate_step_deriv[0].clone())).unwrap();
        
        // Step 1.e: 
        let candidate_tanh_deriv = candidate_tanh.backward(Some(input_gate_candidate_step_deriv[1].clone())).unwrap();
        
        // Step 1.d: 
        let output_gate_sigmoid_deriv = output_gate_sigmoid.backward(Some(new_hidden_state_step_deriv[0].clone())).unwrap();

        // Step 1.c: 
        let input_gate_sigmoid_deriv = input_gate_sigmoid.backward(Some(input_gate_candidate_step_deriv[0].clone())).unwrap();

        // Step 1.b: 
        let forget_gate_sigmoid_deriv = forget_gate_sigmoid.backward(Some(forget_gate_memory_state_step_deriv[0].clone())).unwrap();

        // Step 1.a 
        let mut grad = Matrix::zeros((0,1));
        grad.append(Axis(0), forget_gate_sigmoid_deriv.view()).unwrap();
        grad.append(Axis(0), input_gate_sigmoid_deriv.view()).unwrap();
        grad.append(Axis(0), output_gate_sigmoid_deriv.view()).unwrap();
        grad.append(Axis(0), candidate_tanh_deriv.view()).unwrap();

        // Step 1: 
        let step_1_deriv = step_1.backward(Some(grad)).unwrap();

        (new_hidden_state, new_memory_state, step_1_deriv[0].clone())
    }
}
