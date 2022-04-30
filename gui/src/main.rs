mod data;
use neural_network::{
    Matrix,
    io::IO,
    lotto::LottoNetwork
};
use data::{Encoding};

fn get_all_days_data() -> (Vec<Matrix>, Vec<Vec<Matrix>>) {
    let mut client = Encoding::new();

    let features_labels = client.get_lotto_days_data(&["friday", "monday", "tuesday", "tuesday_night"]);
    let size = features_labels.len();
    let mut new_features: Vec<Matrix> = vec![] ;
    let mut new_labels: Vec<Vec<Matrix>> = vec![];
    
    (0..size).into_iter().for_each(|a| {
        let (input, output) = &features_labels[a];
        new_features.push(input.clone());
        new_labels.push(output.clone());
    });

    (new_features, new_labels)
}

fn _retrain(infos: (&[Matrix], Vec<Vec<Matrix>>), batch_size: usize, epoch: i32, lr: f64, path: &str, dir: &str) {
    let (new_features, new_labels) = infos; 
    let mut loaded_network = LottoNetwork::load(path, dir);
    loaded_network.train(&new_features, new_labels, lr, batch_size, epoch);
    loaded_network.save(dir);
}

fn _initial_training(new_features: &[Matrix], new_labels: Vec<Vec<Matrix>>, h_neurons: usize, h_inputs: usize, s_neurons: usize, s_inputs: usize, batch_size: usize, epoch: i32, lr: f64, dir: &str) {

    let mut nn = LottoNetwork::new(h_neurons, h_inputs, s_neurons, s_inputs);

    nn.train(&new_features, new_labels, lr, batch_size, epoch);
    nn.save(dir);
}

fn predict_for_all(inputs: &[i32], path: &str, dir: &str) {
    // prepare data
    let encoding = Encoding::new();

    let mut loaded_network = LottoNetwork::load(path, dir);
    let inputs_bits =  encoding.get_row_bits_for_featurs(&inputs);

    let predicted = loaded_network.predict(inputs_bits);

    let res = encoding.into_result(&predicted);
    println!("Predicted: {:?}", res);
}


fn main() {
    
    let (_new_features, _new_labels) = get_all_days_data();
    let _batch_size: usize = 10;
    let _epoch = 20000;
    let _lr = 0.003; // lr = 1.0 - momemtum
    let path = "all_lotto_parameters_new_design";
    let dir = "data";
    let _h_neurons = 128;
    let _h_inputs = 90;
    let _s_neurons = 90;
    let _s_inputs = 128;

    //initial_training(&new_features, new_labels, h_neurons, h_inputs, s_neurons, s_inputs, batch_size, epoch, lr, dir);

    //retrain((&new_features, new_labels), batch_size, epoch, lr, path , dir);

    // ####### Prediction Start ##################
    //let inputs = vec![74, 82, 26, 67, 33, 42, 78, 37, 75, 44]; // monday,  win 32 39 33 11 26 Mac 19 58 57 21 31
    //let inputs = [57, 27, 1, 48, 78, 11, 21, 64, 49, 81]; // friday, win 79 61 23 39 64 Mac 11 74 62 51 81
    let inputs = [89, 48, 33, 71, 64, 30, 17, 53, 20, 9]; // tuesday, win 68 47 25 40 36 Mac 86 61 56 1 78
    //let inputs = [77, 23, 89, 90, 59, 58, 15, 29, 24, 87]; // tuesday_night,  win 49 60 17 28 55 Mac 86 13 5 27 87
    // ####### Not in training ####################
    //let inputs = [62, 80, 15, 41, 42, 65, 72, 34, 16, 18]; // friday, win 65 84 25 2 19 Mac 5 67 59 3 39
    //let inputs = [59, 51, 85, 64, 88, 2, 72, 12, 27, 16]; // friday, win 87 18 81 2 76 Mac 62 19 8 25 84
    // ###### End Of Not in training #############
    
    predict_for_all(&inputs, path, dir);
    // ####### Prediction End ##################
}