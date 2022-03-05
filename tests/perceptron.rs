//use djed_maths::linear_algebra::matrix::Matrix;
/*use kongodjan::{
    //maths::types::MatrixD,
    layers::multi_layers::FcLayer,
    activators::non_linear::{logsig, logsig_deriv/*, tansig, tansig_deriv*/},
    network_arch::FCNetwork,
    loss_functions::{squared_loss, squared_loss_gradient},
    neural_traits::NetworkT,
};*/


#[test]
fn full_connected_test() {
/*
    // the first hiden layer initial weights matrix 
    // rows are inputs and columns are neurons
    // 7 inputs and 3 neurons
    // input layer has 7 neurons and first hiden layer has 3 neurons
    let true_w: Matrix<f64> = Matrix::new_from_vec(3, &vec![
        2.0, -3.4, 2.0, 
        2.0, -3.4, 2.0, 
        2.0, -3.4, 2.0, 
        -3.4, 2.0, -3.4, 
        1.0, -3.4, 2.0, 
        -3.4, 1.0, -3.4, 
        2.0, -3.4, 1.0
    ]);

    // the first hiden layer initial bias matrix 
    // one row and 3 columns(neurons)
    let true_b: Matrix<f64> = Matrix::new_from_vec(3, &vec![
        2.0, -3.4, 1.0
    ]);


    // one layer with 3 neuron which receives 7 inputs
    // then features are 7 rows(7 inputs) and more columns
    // labels 3 rows and more columns
    let (features, labels) = synthetic_data_mat(&true_w, true_b, 1000);
    //labels.view();
    
    
    // build layer
    let n_neurons = 4;
    let n_inputs = 7; 
    let n_output_neurons = 3;
    let layer1 = FcLayer::new(n_inputs, n_neurons, logsig, Some(logsig_deriv), 1);
    let layer2 = FcLayer::new(n_neurons, n_output_neurons, logsig, Some(logsig_deriv), 2);

    let layers = vec![layer1, layer2];

    // build the network
    let mut network = FCNetwork::new(features, layers, labels);

    // train the networ
    let _tran = network.train(0.1, Some(5), (squared_loss, squared_loss_gradient,sgd), 1);
    */
/* 
    //let pred = network.predict(&test_x);
    //println!("output: {:?}", pred);
    */
}
