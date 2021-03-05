use rand::{thread_rng, seq::SliceRandom};
use crate::maths::types::MatrixD;
use std::cmp::min;

pub fn data_iter(batch_size: usize, features: &MatrixD<f64>, labels: &MatrixD<f64>) -> Vec<(MatrixD<f64>, MatrixD<f64>)> {
    let mut rng = thread_rng();
    let num_examples = features.ncols();

    let mut indices: Vec<_> = (0..num_examples).into_iter().collect();
    indices.shuffle(&mut rng);

    let mut result = Vec::new();
    for i in (0..num_examples).into_iter().step_by(batch_size){
        let batch_indices = indices.as_slice()[i..min(i + batch_size, num_examples)].to_vec();
        
        let mut feat = Vec::new();
        let mut lab = Vec::new();
        batch_indices.iter().for_each(|a| {
            feat.push(features.column(*a));
            lab.push(labels.column(*a));
        });

        let new_features = MatrixD::from_columns(feat.as_slice());
        let new_labels = MatrixD::from_columns(lab.as_slice());

        result.push((new_features, new_labels));
    }

    result
} 