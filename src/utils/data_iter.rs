use random_number::rand::{thread_rng, seq::SliceRandom};
use djed_maths::linear_algebra::matrix::Matrix;
use std::cmp::min;

pub fn data_iter(batch_size: usize, features: &Matrix<f64>, labels: &Matrix<f64>) -> Vec<(Matrix<f64>, Matrix<f64>)> {
    let mut rng = thread_rng();
    let num_examples = features.ncols();

    let mut indices: Vec<_> = (0..num_examples).into_iter().collect();
    indices.shuffle(&mut rng);

    let mut result = Vec::new();
    for i in (0..num_examples).into_iter().step_by(batch_size) {
        let batch_indices = indices.as_slice()[i..min(i + batch_size, num_examples)].to_vec();
        
        let mut feat = Vec::new();
        let mut lab = Vec::new();

        batch_indices.iter().for_each(|a| {
            feat.push(features.get_col(*a));
            lab.push(labels.get_col(*a));
        });
        
        let new_features = Matrix::<f64>::new_from_columns(&feat);
        let new_labels = Matrix::<f64>::new_from_columns(&lab);
        
        result.push((new_features, new_labels));

    }

    result

} 
