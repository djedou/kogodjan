use neural_network::{Matrix};
use super::{LottoClient, CounterData};
use std::cmp::Ordering;
pub struct Encoding;
use std::collections::HashMap;

impl Encoding {
    pub fn new() -> Self {Encoding}

    pub fn _positional_encoding(position: f64, dimension: usize, pe: &mut [f64]) {
        for i in 0..dimension {
            let ex = ((2 * i) / dimension) as f64;
            let h: f64 = 100.;

            pe[i] = (position / h.powf(ex)).sin();
            pe[i + 1] = (position / h.powf(ex)).cos();
        }
    }

    pub fn get_lotto_days_data_winners(&mut self, days: &[&str]) -> Vec<(Matrix, Vec<Matrix>)> {

        let mut data: Vec<(Matrix, Vec<Matrix>)> = vec![];

        for day in days {
            let mut res = self.get_one_day_data_winners(day);
            data.append(&mut res);
        }

        data
    }

    pub fn get_lotto_days_data_counterparts(&mut self, days: &[&str]) -> Vec<(Matrix, Vec<Matrix>)> {

        let mut data: Vec<(Matrix, Vec<Matrix>)> = vec![];
        let counter = LottoClient::new().get_counter_data("counterpart");

        for day in days {
            let mut res = self.get_one_day_data_counterparts(day, &counter);
            data.append(&mut res);
        }

        data
    }

    pub fn get_one_day_data_winners(&mut self, day: &str) -> Vec<(Matrix, Vec<Matrix>)> {
        let mut client = LottoClient::new();
        
        let data = client.get_game_data(day);

        let mut features_labels: Vec<(Matrix, Vec<Matrix>)> = Vec::new();

        let data_len = data.len();
        for n in 1..data_len  {
            if n == data_len {
                break;
            }
            
            let input_event = data.get(&(n as i32));
            let output_event = data.get(&((n + 1) as i32));
            if let Some(input) = input_event {
                if let Some(output) = output_event {
                    let input_mat = self.get_row_bits_for_features_winners(&[input.w_1, input.w_2, input.w_3, input.w_4, input.w_5, input.m_1, input.m_2, input.m_3, input.m_4, input.m_5]);
                    
                    let output_mat = self.get_row_bits_for_label(&[output.w_1, output.w_2, output.w_3, output.w_4, output.w_5, output.m_1, output.m_2, output.m_3, output.m_4, output.m_5]);

                    features_labels.push((input_mat, output_mat));
                }
            }
        }

        features_labels
    }

    pub fn get_one_day_data_counterparts(&mut self, day: &str, counter: &HashMap<i32, CounterData>) -> Vec<(Matrix, Vec<Matrix>)> {
        let mut client = LottoClient::new();
        
        let data = client.get_game_data(day);

        let mut features_labels: Vec<(Matrix, Vec<Matrix>)> = Vec::new();

        let data_len = data.len();
        for n in 1..data_len  {
            if n == data_len {
                break;
            }
            
            let input_event = data.get(&(n as i32));
            let output_event = data.get(&((n + 1) as i32));
            if let Some(input) = input_event {
                if let Some(output) = output_event {
                    let input_mat = self.get_row_bits_for_features_counterpats(&[input.w_1, input.w_2, input.w_3, input.w_4, input.w_5, input.m_1, input.m_2, input.m_3, input.m_4, input.m_5], &counter);
                    
                    let output_mat = self.get_row_bits_for_label(&[output.w_1, output.w_2, output.w_3, output.w_4, output.w_5, output.m_1, output.m_2, output.m_3, output.m_4, output.m_5]);

                    features_labels.push((input_mat, output_mat));
                }
            }
        }

        features_labels
    }

    pub fn get_row_bits_for_features_winners(&self, row: &[i32]) -> Matrix {

        let mut values: [f64; 90] = [0.; 90];
        
        for (_i,n) in row.iter().enumerate() {
            values[(n - 1) as usize] = 1.0;
        }
        Matrix::from_shape_vec((values.len(), 1), values.to_vec()).unwrap()
    }

    pub fn get_row_bits_for_features_counterpats(&self, row: &[i32], counter: &HashMap<i32, CounterData>) -> Matrix {

        let mut values: [f64; 90] = [0.; 90];
        
        for (_i,n) in row.iter().enumerate() {
            
            if let Some(CounterData {number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code }) = counter.get(&n) {
                if *number >= 1 && *number <= 90 {}
                if *counter >= 1 && *counter <= 90 {
                    values[(counter - 1) as usize] = 1.0;
                }
                if *bonanza >= 1 && *bonanza <= 90 {
                    values[(bonanza - 1) as usize] = 1.0;
                }
                if *stringkey >= 1 && *stringkey <= 90 {
                    values[(stringkey - 1) as usize] = 1.0;
                }
                if *turning >= 1 && *turning <= 90 {
                    values[(turning - 1) as usize] = 1.0;
                }
                if *melta >= 1 && *melta <= 90 {
                    values[(melta - 1) as usize] = 1.0;
                }
                if *partner >= 1 && *partner <= 90 {
                    values[(partner - 1) as usize] = 1.0;
                }
                if *equivalent >= 1 && *equivalent <= 90 {
                    values[(equivalent - 1) as usize] = 1.0;
                }
                if *shadow >= 1 && *shadow <= 90 {
                    values[(shadow - 1) as usize] = 1.0;
                }
                if *code >= 1 && *code <= 90 {
                    values[(code - 1) as usize] = 1.0;
                }
            }
        }
        Matrix::from_shape_vec((values.len(), 1), values.to_vec()).unwrap()
    }

    pub fn get_row_bits_for_label(&self, row: &[i32]) -> Vec<Matrix> {

        let mut values: Vec<Matrix> = vec![];
        
        for (_i,n) in row.iter().enumerate() {
            let mut res: [f64; 90] = [0.; 90];
            res[(n - 1) as usize] = 1.0;
            values.push(Matrix::from_shape_vec((res.len(), 1), res.to_vec()).unwrap());
        }

        values
    }

    pub fn into_result(&self, outputs: &[Matrix]) -> Vec<i32> {

        let mut res: Vec<i32> = vec![];
        
        for o in outputs {
            let o_vec = o.column(0).to_vec();
            res.push(self.get_max_index(&o_vec));
        }
        res
    }

    fn get_max_index(&self, nets: &[f64]) -> i32 {
        
        let index_of_max: Option<usize> = nets
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            .map(|(index, _)| index);

        index_of_max.unwrap() as i32 + 1
    }
}