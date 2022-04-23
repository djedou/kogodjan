use neural_network::{Matrix};
use super::{LottoClient, CounterData, Bits};
use std::collections::{HashMap};
pub struct Encoding;


impl Encoding {
    pub fn new() -> Self {Encoding}

    pub fn positional_encoding(position: f64, dimension: usize, pe: &mut [f64]) {
        for i in 0..dimension {
            let ex = ((2 * i) / dimension) as f64;
            let h: f64 = 100.;

            pe[i] = (position / h.powf(ex)).sin();
            pe[i + 1] = (position / h.powf(ex)).cos();
        }
    }

    pub fn get_lotto_days_data(&mut self, days: &[&str]) -> Vec<(Matrix, Matrix)> {
        let mut client = LottoClient::new();

        let mut data: Vec<(Matrix, Matrix)> = vec![];
        let counter = client.get_counter_data("counterpart");

        for day in days {
            let mut res = self.get_one_day_data(day, &counter);
            data.append(&mut res);
        }

        data
    }

    pub fn get_one_day_data(&mut self, day: &str, counter: &HashMap<i32, CounterData>) -> Vec<(Matrix, Matrix)> {
        let mut client = LottoClient::new();
        
        let data = client.get_game_data(day);

        let mut features_labels: Vec<(Matrix, Matrix)> = Vec::new();

        let data_len = data.len();
        for n in 1..data_len  {
            if n == data_len {
                break;
            }
            
            let input_event = data.get(&(n as i32));
            let output_event = data.get(&((n + 1) as i32));
            if let Some(input) = input_event {
                if let Some(output) = output_event {
                    let input_vec = self.get_row_bits(&[input.w_1, input.w_2, input.w_3, input.w_4, input.w_5, input.m_1, input.m_2, input.m_3, input.m_4, input.m_5], &counter);
                    let input_matrix = Matrix::from_shape_vec((input_vec.len(), 1), input_vec).unwrap();
                    
                    let output_vec = self.get_row_bits(&[output.w_1, output.w_2, output.w_3, output.w_4, output.w_5, output.m_1, output.m_2, output.m_3, output.m_4, output.m_5], &counter);
                    let outputs = Matrix::from_shape_vec((output_vec.len(), 1), output_vec).unwrap();
                    
                    features_labels.push((input_matrix, outputs));
                }
            }
        }

        features_labels
    }
    
    pub fn get_row_bits(&self, row: &[i32], counter: &HashMap<i32, CounterData>) -> Vec<f64> {
        
        let mut res: Vec<f64> = vec![];
        
        for (_i,n) in row.iter().enumerate() {

            let mut values: [i32; 9] = [0; 9];
            if let Some(CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code}) = counter.get(&n) {
                if *number <= 90 && *number >= 1 {}
                if *counter <= 90 && *counter >= 1 {
                    values[0] = *counter;
                }
                if *bonanza <= 90 && *bonanza >= 1 {
                    values[1] = *bonanza;
                }
                if *stringkey <= 90 && *stringkey >= 1 {
                    values[2] = *stringkey;
                }
                if *turning <= 90 && *turning >= 1 {
                    values[3] = *turning;
                }
                if *melta <= 90 && *melta >= 1 {
                    values[4] = *melta;
                }
                if *partner <= 90 && *partner >= 1 {
                    values[5] = *partner;
                }
                if *equivalent <= 90 && *equivalent >= 1 {
                    values[6] = *equivalent;
                }
                if *shadow <= 90 && *shadow >= 1 {
                    values[7] = *shadow;
                }
                if *code <= 90 && *code >= 1 {
                    values[8] = *code;
                }
            }
            
            for r in values {
                res.extend_from_slice(&Bits::int_into_bits(r));
            }
        }

        res
    }

    pub fn into_result(&mut self, row: &[f64], counter: &HashMap<i32, CounterData>) -> Vec<i32> {

        let mut chuncks: Vec<Vec<f64>> = vec![];
        for r in row.chunks(63) {
            chuncks.push(r.into());
        }

        let mut res: Vec<i32> = vec![];
        for (i, v) in chuncks.iter().enumerate() {
            let mut chcks: Vec<i32> = vec![];
            for r in v.chunks(7) {
                let m_bits = Bits::float_into_bits(r);
                let m = Bits::bits_into_integer(&m_bits) as i32;
                chcks.push(m);
            }
            let win = self.winner(i, &chcks, &counter);
            res.push(win);
        }

        res
    }

    
    fn winner(&mut self, index: usize, candidate: &[i32], counter: &HashMap<i32, CounterData>) -> i32 {
        
        let mut win = 0;
        let mut wins = vec![];
        for (_, c) in counter {
            let mut score = 0;
            let CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code} = c;
            // Counter
            if candidate[0] == *counter {
                score = score + 1;
            }
            // Bonanza
            if candidate[1] == *bonanza  {
                score = score + 1;
            }
            // Stringkey
            if candidate[2] == *stringkey  {
                score = score + 1;
            }
            // Turning
            if candidate[3] == *turning  {
                score = score + 1;
            }
            // Melta
            if candidate[4] == *melta  {
                score = score + 1;
            }
            // Partner
            if candidate[5] == *partner  {
                score = score + 1;
            }
            // Equivalent
            if candidate[6] == *equivalent  {
                score = score + 1;
            }
            // Shadow
            if candidate[7] == *shadow  {
                score = score + 1;
            }
            // Code
            if candidate[8] == *code  {
                score = score + 1;
            }

            if score >= 2 {
                wins.push(number);
                win = *number;
            }
        }
        println!("candiates: {:?} => {:?}", index + 1, wins);

        win
    }
}