use random_number::rand::{thread_rng, seq::SliceRandom};
use std::cmp::min;
use ndarray::{Array2};

pub fn data_iter(batch_size: usize, features: &Array2<f64>, labels: &Array2<f64>) -> Vec<(Array2<f64>, Array2<f64>)> {
    let mut rng = thread_rng();
    let num_examples = features.ncols();
    let features_rows = features.nrows();
    let labels_rows = labels.nrows();

    let mut indices: Vec<_> = (0..num_examples).into_iter().collect();
    indices.shuffle(&mut rng);

    let mut result = Vec::new();
    
    for i in (0..num_examples).into_iter().step_by(batch_size) {
        let batch_indices = indices.as_slice()[i..min(i + batch_size, num_examples)].to_vec();

        let mut feat = Array2::zeros((features_rows, 0));
        let mut lab = Array2::zeros((labels_rows, 0));

        batch_indices.iter().for_each(|a| {
            feat.push_column(features.column(*a)).unwrap();
            lab.push_column(labels.column(*a)).unwrap();
        });
        
        let new_features = feat;
        let new_labels = lab;
        
        result.push((new_features, new_labels));

    }

    result

} 
