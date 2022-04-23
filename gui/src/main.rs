mod data;
use neural_network::{
    networks::FeedForward,
    Matrix,
    networks::NetworkT,
    Activator,
    Loss,
    io::IO
};
use data::{LottoClient, Encoding, Bits};

fn testing() {
    let mut pe = vec![0., 0., 0., 0., 0., 0., 0., 0., 0.];

    Encoding::positional_encoding(2., pe.len() - 1, &mut pe);
    println!("p_encoding: {:#?}", pe);

    let bits = Bits::int_into_bits(12);
    println!("bits: {:#?}", bits);
}

fn get_all_days_data() -> (usize, usize, Vec<Matrix>, Vec<Matrix>) {
    // prepare data
    //let mut client = LottoClient::new();
    //let counter = client.get_counter_data("counterpart");
    let mut encoding = Encoding::new();

    //let features_labels = client.get_lotto_days_data(&["friday", "monday", "tuesday", "tuesday_night"]);
    let features_labels = encoding.get_lotto_days_data(&["friday", "monday", "tuesday", "tuesday_night"]);
    //let test = features_labels[0].clone();
    //let res = encoding.into_result(test.0.column(0).as_slice().unwrap(), &counter);
    //println!("total: {:#?}", features_labels.len());
    //println!("res: {:?}", res);
    let size = features_labels.len();
    let mut new_features: Vec<Matrix> = vec![] ;
    let mut new_labels: Vec<Matrix> = vec![];
    
    (0..size).into_iter().for_each(|a| {
        let (input, output) = &features_labels[a];
        new_features.push(input.clone());
        new_labels.push(output.clone());
    });
    let feat_nrows = new_features[0].nrows();
    let lab_nrows = new_labels[0].nrows();

    (feat_nrows, lab_nrows, new_features, new_labels)
}

/*
fn get_counter_data() -> (usize, usize, Vec<Matrix>, Vec<Matrix>) {
    // prepare data
    let mut client = LottoClient::new();
    
    // ######## Counter Part Netork Data Start #################
    let (features, labels) = client.get_counter_part_network();
    let size = features.len();
    let feat_nrows = features[0].len();
    let lab_nrows = labels[0].len();
    
    let mut new_features: Vec<Matrix> = vec![] ;
    let mut new_labels: Vec<Matrix> = vec![];
    
    (0..size).into_iter().for_each(|a| {
        new_features.push(Matrix::from_shape_vec((feat_nrows, 1), features[a].clone()).unwrap());
        new_labels.push(Matrix::from_shape_vec((lab_nrows, 1), labels[a].clone()).unwrap());
    });
   
    (feat_nrows, lab_nrows, new_features, new_labels)
}
*/

fn initial_training(infos: (usize, usize, &[Matrix], &[Matrix]), batch_size: usize, epoch: i32, lr: f64, momemtum: f64, dir: &str) {
    let (feat_nrows, lab_nrows, new_features, new_labels) = infos; 

    let hidden_layer_neurons = 1024; //128; //256;
    let mut nn = FeedForward::new(&[(feat_nrows, hidden_layer_neurons, Activator::Sigmoid), (hidden_layer_neurons, lab_nrows, Activator::Sigmoid)], Loss::Mse);

    nn.train(&new_features, &new_labels, lr, momemtum, batch_size, epoch);
    nn.save(dir);
}


fn retrain(infos: (&[Matrix], &[Matrix]), batch_size: usize, epoch: i32, lr: f64, momemtum: f64, path: &str, dir: &str) {
    let (new_features, new_labels) = infos; 
    let mut loaded_network = FeedForward::load(path, dir);
    loaded_network.train(&new_features, &new_labels, lr, momemtum, batch_size, epoch);
    loaded_network.save(dir);
}

/*
fn prediction(input: Matrix, path: &str, dir: &str) {
    // ###### Prediction Start ########
    let mut loaded_network = FeedForward::load(path, dir);
    let predicted = loaded_network.predict(&input);
    

    let mut res = vec![];
    for ch in predicted.column(0).to_vec().chunks(7) {

        let mut ch_bits = vec![];
        for c in ch {
            if *c >= 0.5 {
                ch_bits.push(1 as u8);
            }
            else {
                ch_bits.push(0 as u8);
            }
        }

        res.push(ch_bits);
    }

    let feat_bits = into_bits(&input.column(0).to_vec());
    let feat_int = into_int(&feat_bits);
    println!("input: {}", feat_int);
    println!("output:  ");
    for (i,v) in res.iter().enumerate() {
        let dec = into_int(&v.as_slice());
        println!("{} : {:?} ==> {}",i+1,v, dec);
    }
}
*/

fn predict_for_all(inputs: &[i32], path: &str, dir: &str) {
    // prepare data
    let mut client = LottoClient::new();
    let mut encoding = Encoding::new();

    let mut loaded_network = FeedForward::load(path, dir);
    let counter = client.get_counter_data("counterpart");
    let inputs_bits =  encoding.get_row_bits(&inputs, &counter);

    let inputs_mat = Matrix::from_shape_vec((inputs_bits.len(), 1), inputs_bits).unwrap();
    let predicted = loaded_network.predict(&inputs_mat);

    let res = encoding.into_result(&predicted.column(0).to_vec(), &counter);
    println!("Predicted: {:?}", res);
}

/*
fn into_int(bits: &[u8]) -> u8 {
    bits.iter()
        .fold(0, |result, &bit| {
            (result << 1) ^ bit
        })
}

fn into_bits(ch: &[f64]) -> Vec<u8> {
    let mut ch_bits = vec![];
    for c in ch {
        if *c > 0.49 {
            ch_bits.push(1 as u8);
        }
        else {
            ch_bits.push(0 as u8);
        }
    }
    ch_bits
}
*/

fn main() {
    
    //let (feat_nrows, lab_nrows, new_features, new_labels) = get_counter_data();
    let (_feat_nrows, _lab_nrows, new_features, new_labels) = get_all_days_data();
    //println!("feat_nrows : {:#?}", feat_nrows);
    //println!("lab_nrows : {:#?}", lab_nrows);
    let batch_size: usize = 630;
    let epoch = 8000;
    let lr = 0.003; // lr = 1.0 - momemtum
    let momemtum = 0.997; 
    let path = "all_lotto_parameters_new_design";
    let dir = "data";

    //initial_training((feat_nrows, lab_nrows, &new_features, &new_labels), batch_size, epoch, lr, momemtum, dir);
    
    retrain((&new_features, &new_labels), batch_size, epoch, lr, momemtum, path , dir);
   
    // ####### Prediction Start ##################
    //let inputs = vec![74, 82, 26, 67, 33, 42, 78, 37, 75, 44]; // monday,  win 32 39 33 11 26 Mac 19 58 57 21 31
    //let inputs = [57, 27, 1, 48, 78, 11, 21, 64, 49, 81]; // friday, win 79 61 23 39 64 Mac 11 74 62 51 81
    // ####### Not in training ####################
    //let inputs = [62, 80, 15, 41, 42, 65, 72, 34, 16, 18]; // friday, win 65 84 25 2 19 Mac 5 67 59 3 39
    //let inputs = [59, 51, 85, 64, 88, 2, 72, 12, 27, 16]; // friday, win 87 18 81 2 76 Mac 62 19 8 25 84
    // ###### End Of Not in training #############
    
    //predict_for_all(&inputs, path, dir);
    // ####### Prediction End ##################
}



    /*
    // ##################### Building and Initial Training Start Here ############################
    // build layers
    let network_input = new_features[0].nrows();
    let network_output_neurons = new_labels[0].nrows(); 
    let hidden_layer_neurons = 1024;
    
    // build the network
    let mut network = FeedForward::new(
        &[
            (network_input, None), 
            (hidden_layer_neurons, Some(Activation::Sigmoid)),
            (hidden_layer_neurons, Some(Activation::Sigmoid)), 
            (network_output_neurons, Some(Activation::Sigmoid))
        ],
        Loss::Mse
    );
    
    // train the network
    let batch_size: usize = 10;
    let epoch = 100;
    
    network.train(&new_features, &new_labels, 0.003, batch_size, epoch); 
    
    network.save("data");
    // ##################### Building and Initial Training End Here ############################
    */
     /*
    // ##################### Training Start Here ############################
    let mut loaded_network = FeedForward::load("all_lotto_parameters", "data");
    // retrain the networ
    let batch_size: usize = 10;
    let epoch = 500;
    loaded_network.train(&new_features, &new_labels, 0.003, batch_size, epoch); 
    
    loaded_network.save("data");
    // ##################### Training End Here ############################
    */
    /*
      

*/