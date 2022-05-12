mod data;
use neural_network::{
    Matrix,
    io::IO,
    lotto::LottoNetwork
};
use data::{Encoding, LottoClient};


fn get_all_days_data_winners() -> (Vec<Matrix>, Vec<Vec<Matrix>>) {
    let mut client = Encoding::new();

    let features_labels = client.get_lotto_days_data_winners(&["friday", "monday", "tuesday", "tuesday_night", "mid_week"]);
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

fn get_all_days_data_counterparts() -> (Vec<Matrix>, Vec<Vec<Matrix>>) {
    let mut client = Encoding::new();

    let features_labels = client.get_lotto_days_data_counterparts(&["friday", "monday", "tuesday", "tuesday_night", "mid_week"]);
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

fn retrain(infos: (&[Matrix], Vec<Vec<Matrix>>), batch_size: usize, epoch: i32, lr: f64, path: &str, dir: &str, breakpoint: f64) {
    let (new_features, new_labels) = infos; 
    let mut loaded_network = LottoNetwork::load(path, dir);
    loaded_network.train(&new_features, new_labels, lr, batch_size, epoch, breakpoint);
    loaded_network.save(dir);
}

fn initial_training(new_features: &[Matrix], new_labels: Vec<Vec<Matrix>>, h_neurons: usize, h_inputs: usize, s_neurons: usize, s_inputs: usize, batch_size: usize, epoch: i32, lr: f64, dir: &str, breakpoint: f64) {

    let mut nn = LottoNetwork::new(h_neurons, h_inputs, s_neurons, s_inputs);

    nn.train(&new_features, new_labels, lr, batch_size, epoch, breakpoint);
    nn.save(dir);
}

fn predict_for_winners(inputs: &[i32], path: &str, dir: &str) {
    // prepare data
    let encoding = Encoding::new();

    let mut loaded_network = LottoNetwork::load(path, dir);
    let inputs_bits =  encoding.get_row_bits_for_features_winners(&inputs);

    let predicted = loaded_network.predict(inputs_bits);

    let res = encoding.into_result(&predicted);
    println!("Predicted: {:?}", res);
}

fn predict_for_counterparts(inputs: &[i32], path: &str, dir: &str) {
    // prepare data
    let encoding = Encoding::new();
    let counter = LottoClient::new().get_counter_data("counterpart");


    let mut loaded_network = LottoNetwork::load(path, dir);
    let inputs_bits =  encoding.get_row_bits_for_features_counterpats(&inputs, &counter);

    let predicted = loaded_network.predict(inputs_bits);

    let res = encoding.into_result(&predicted);
    println!("Predicted: {:?}", res);
}


fn main() {
    
    //let (new_features, new_labels) = get_all_days_data_winners();
    let (new_features, new_labels) = get_all_days_data_counterparts();
    let batch_size: usize = 900;
    let epoch = 10000; //20000;
    let lr = 0.01;
    //let path = "winners_parameters";
    let path = "counterparts_parameters_all_256";
    let dir = "data";
    let h_neurons = 180;
    let h_inputs = 90;
    let s_neurons = 90;
    let s_inputs = 180;
    let breakpoint = 0.008;

    initial_training(&new_features, new_labels, h_neurons, h_inputs, s_neurons, s_inputs, batch_size, epoch, lr, dir, breakpoint);

    //retrain((&new_features, new_labels), batch_size, epoch, lr, path , dir, breakpoint);

    // ####### Prediction Start ##################
    //let inputs = vec![74, 82, 26, 67, 33, 42, 78, 37, 75, 44]; // monday,  win 32 39 33 11 26 Mac 19 58 57 21 31
    //let inputs = [57, 27, 1, 48, 78, 11, 21, 64, 49, 81]; // friday, win 79 61 23 39 64 Mac 11 74 62 51 81
    //let inputs = [89, 48, 33, 71, 64, 30, 17, 53, 20, 9]; // tuesday, win 68 47 25 40 36 Mac 86 61 56 1 78
    //let inputs = [77, 23, 89, 90, 59, 58, 15, 29, 24, 87]; // tuesday_night,  win 49 60 17 28 55 Mac 86 13 5 27 87
    //let inputs = [25, 64, 75, 16, 76, 29, 63, 84, 42, 68]; // mid_weed, Win 26 46 72 84 74 Mac 4 24 69 77 38
    // ####### Not in training ####################
    //let inputs = [62, 80, 15, 41, 42, 65, 72, 34, 16, 18]; // friday, win 65 84 25 2 19 Mac 5 67 59 3 39
    //let inputs = [59, 51, 85, 64, 88, 2, 72, 12, 27, 16]; // friday, win 87 18 81 2 76 Mac 62 19 8 25 84
    // ###### End Of Not in training #############
    //let inputs = [90, 61, 80, 72, 84, 16, 11, 2, 23, 10]; // friday, win
    //predict_for_winners(&inputs, path, dir);
    //predict_for_counterparts(&inputs, path, dir);
    // ####### Prediction End ##################
}