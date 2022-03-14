use random_number::rand::{thread_rng, seq::SliceRandom};
use std::cmp::min;
use ndarray::{Array2};

pub fn data_iter(batch_size: usize, features: &[Array2<f64>], labels: &[Array2<f64>]) -> Vec<(Vec<Array2<f64>>, Vec<Array2<f64>>)> {
    let mut rng = thread_rng();
    let num_examples = features.len();

    let mut indices: Vec<_> = (0..num_examples).into_iter().collect();
    indices.shuffle(&mut rng);

    let mut result = Vec::new();
    
    for i in (0..num_examples).into_iter().step_by(batch_size) {
        let batch_indices = indices.as_slice()[i..min(i + batch_size, num_examples)].to_vec();

        let mut feat: Vec<Array2<f64>> = vec![];
        let mut lab: Vec<Array2<f64>> = vec![];

        batch_indices.iter().for_each(|a| {
            feat.push(features[*a].clone());
            lab.push(labels[*a].clone());
        });
        
        result.push((feat, lab));
    }

    result
} 
