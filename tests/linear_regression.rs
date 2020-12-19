
use kongodjan::{
    maths::types::MatrixD,
    utils::{synthetic_data},
    layers::single_layer::LrLayer,
    activators::linear::purelin,
    network_arch::LinearNetwork,
    loss_functions::{squared_loss, squared_loss_gradient},
    neural_traits::NetworkT,
    optimizers::sgd
};


#[test]
fn linear_regression() {

    // rows are neurons and columns are inputs
    let true_w = MatrixD::<f32>::from_row_slice(1, 2, &[
        2.0, -3.4
    ]);

    let true_b = 4.2;
    
    // the network is one layer with one neuron which receives two inputs
    // then features are two rows(two inputs) and more columns
    // labels one row and more columns
    let (features, labels) = synthetic_data(&true_w, true_b, 1000);

    // build layer
    let size_of_input_vec = 2; 
    let layer = LrLayer::new(size_of_input_vec, purelin, None, 1);
    
    // build the network
    let mut network = LinearNetwork::new(features, layer, labels);

    // train the networ
    network.train(0.01, Some(5), (squared_loss, squared_loss_gradient,sgd), 20);

    //let pred = network.predict(&test_x);
    //println!("output: {:?}", pred);
}
