use rand::{thread_rng, seq::SliceRandom};
use std::cmp::min;
use algo_diff::maths::Matrix;

pub fn data_iter(batch_size: usize, features: &[Matrix], labels: &[Matrix]) -> Vec<(Matrix, Matrix)> {
    let mut rng = thread_rng();
    let num_examples = features.len();
    let feat_nrows = features[0].nrows();
    let lab_nrows = labels[0].nrows();
    let mut indices: Vec<_> = (0..num_examples).into_iter().collect();
    indices.shuffle(&mut rng);

    let mut result = Vec::new();
    
    for i in (0..num_examples).into_iter().step_by(batch_size) {
        let batch_indices = indices.as_slice()[i..min(i + batch_size, num_examples)].to_vec();

        let mut feat = Matrix::zeros((feat_nrows, 0));
        let mut lab = Matrix::zeros((lab_nrows, 0));

        batch_indices.iter().for_each(|a| {
            feat.push_column(features[*a].column(0)).unwrap();
            lab.push_column(labels[*a].column(0)).unwrap();
        });
        
        result.push((feat, lab));
    }

    result
} 
