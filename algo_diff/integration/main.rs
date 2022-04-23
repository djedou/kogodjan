use algo_diff::{
    graphs::{LinearGraph, Graph, MseGraph, SigmoidGraph},
    maths::Matrix
};

fn forward_backward() {
    
    // Forward 
    // Step 1: WX + B
    let mut linear_graph = LinearGraph::new();
    let x = Matrix::from_shape_vec((4,1), vec![1., 2., 1., 0.]).unwrap();
    let w = Matrix::from_shape_vec((4,2), vec![1., 0., 0., -1., 1., -1., 1., -2.]).unwrap();
    let w_t = w.clone().reversed_axes();
    let b = Matrix::from_shape_vec((2,1), vec![1., 1.]).unwrap();
    
    let wx_b_o = linear_graph.forward([w_t, x.clone(), b]);
    
    // Step 2 : Activatio function
    let mut sigmoid_graph = SigmoidGraph::new();
    
    let sig_o = sigmoid_graph.forward(wx_b_o);
    println!("Output: {:#?}", sig_o);
    println!(" ");
    
    // Step 3: Loss
    let mut mse_graph = MseGraph::new();
    let t = Matrix::from_shape_vec((2,1), vec![1., 1.]).unwrap();
    
    let loss = mse_graph.forward([t, sig_o]);
    println!("loss: {:#?}", loss);
    println!(" ");
    
    // Backward 
    // Step 1
    let loss_d = mse_graph.backward(None).unwrap();
    
    // Step 2
    let sig_d = sigmoid_graph.backward(Some(loss_d)).unwrap();

    // Step 3
    let wx_b_d = linear_graph.backward(Some(sig_d));
    println!("weights: {:#?}", w);
    println!("inputs: {:#?}", x);
    println!("Gradients: {:#?}", wx_b_d);
    println!(" ");
}



fn main() {
    forward_backward();
 }