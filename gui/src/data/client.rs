use postgres::{
    Client, 
    NoTls,
    row::Row,
    Error
};
use crate::data::{CounterData, LottoData};
use std::collections::{HashMap};
use neural_network::{Matrix};

pub struct LottoClient {
    client: Result<Client, Error> 
}

impl LottoClient {
    pub fn new() -> Self {
        let client = Client::connect("host=localhost user=postgres password=djedou dbname=postgres", NoTls);

        LottoClient {
            client
        }
    }

    pub fn get_game_data(&mut self, name: &str) -> HashMap<i32, LottoData> {

        let query = format!("SELECT * FROM lotto.{:?}",name);

        let mut data: HashMap<i32, LottoData> = HashMap::new();

        if let Ok(ref mut cl) = self.client {
            for rows in cl.query(&query[..], &[]) {
                rows.iter().for_each(|r| {
                    let item = self.get_row(r);
                    data.insert(item.event, item);
                });
            }
        }

        data
    }

    fn get_row(&self, row: &Row) -> LottoData {
        LottoData {
            event: row.get("lottoevent"),
            w_1: row.get("win1"),
            w_2: row.get("win2"),
            w_3: row.get("win3"),
            w_4: row.get("win4"),
            w_5: row.get("win5"),
            m_1: row.get("mac1"),
            m_2: row.get("mac2"),
            m_3: row.get("mac3"),
            m_4: row.get("mac4"),
            m_5: row.get("mac5")
        }
    }

    pub fn get_counter_data(&mut self, counter: &str) -> HashMap<i32 ,CounterData> {
    
        let query = format!("SELECT * FROM lotto.{:?}", counter);
    
        let mut data: HashMap<i32 ,CounterData> = HashMap::new();
    
        if let Ok(ref mut cl) = self.client {
            for rows in cl.query(&query[..], &[]) {
                rows.iter().for_each(|r| {
                    let item = self.get_counter_row(r);
                    data.insert(item.number, item);
                });
            }
        }
        
        data
    }

    fn get_counter_row(&self, row: &Row) -> CounterData {
        CounterData {
            number: row.get("numbers"),
            counter: row.get("counter"),
            bonanza: row.get("bonanza"),
            stringkey: row.get("stringkey"),
            turning: row.get("turning"),
            melta: row.get("melta"),
            partner: row.get("partner"),
            equivalent: row.get("equivalent"),
            shadow: row.get("shadow"),
            code: row.get("code")
        }
    }


    /*
    pub fn _get_counter_part_vec(&mut self, counter: &str) -> Vec<CounterData> {
    
        let query = format!("SELECT * FROM lotto.{:?} order by lottoevnt", counter);
    
        let mut data: Vec<CounterData> = Vec::new();
    
        if let Ok(ref mut cl) = self.client {
            for rows in cl.query(&query[..], &[]) {
                rows.iter().for_each(|r| {
                    let item = self.get_counter_row(r);
                    data.push(item);
                });
            }
        }
        
        data
    }

    pub fn _get_lotto_data(&mut self, size: usize, day: &str) -> Vec<(Vec<Matrix>, Matrix)> {
        let data = self.get_game_data(day);
        let counter = self.get_counter_data("counterpart");

        let mut features_labels: Vec<(Vec<Matrix>, Matrix)> = Vec::new();

        let data_len = data.len() - 1;
        println!("data_len: {:?}", data_len);
        for n in 1..data_len  {
            // input is here
            let mut inputs: Vec<Matrix> = vec![];
            if data_len < (n + size) {
                break;
            }
            
            for i in 0..size {
                let event = data.get(&((n + i) as i32)).unwrap();
                let res = self.get_row_bits(&[event.w_1, event.w_2, event.w_3, event.w_4, event.w_5, event.m_1, event.m_2, event.m_3, event.m_4, event.m_5], &counter);
                inputs.push(Array::from_shape_vec((res.len(), 1), res).unwrap());
            }

            let output_event = data.get(&((n + size) as i32)).unwrap();
            let res = self.get_row_bits(&[output_event.w_1, output_event.w_2, output_event.w_3, output_event.w_4, output_event.w_5, output_event.m_1, output_event.m_2, output_event.m_3, output_event.m_4, output_event.m_5], &counter);
            let outputs = Array::from_shape_vec((res.len(), 1), res).unwrap();
            

            features_labels.push((inputs, outputs));

            break;
        }

        features_labels
    }
*/

    pub fn get_row_bits(&self, row: &[i32], counter: &HashMap<i32, CounterData>) -> Vec<f64> {
        let mut values: [f64; 90] = [0.; 90];
        let mut win_mac: [f64; 90] = [0.; 90];
        
        for (i,n) in row.iter().enumerate() {
            if let Some(CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code}) = counter.get(&n) {
                if *number <= 90 && *number >= 1 {
                    win_mac[(*number - 1) as usize] = if i < 5 {1.0} else {0.};
                }
                if *counter <= 90 && *counter >= 1 {
                    values[(*counter - 1) as usize] = 1.0;
                }
                if *bonanza <= 90 && *bonanza >= 1 {
                    values[(*bonanza - 1) as usize] = 1.0;
                }
                if *stringkey <= 90 && *stringkey >= 1 {
                    values[(*stringkey - 1) as usize] = 1.0;
                }
                if *turning <= 90 && *turning >= 1 {
                    values[(*turning - 1) as usize] = 1.0;
                }
                if *melta <= 90 && *melta >= 1 {
                    values[(*melta - 1) as usize] = 1.0;
                }
                if *partner <= 90 && *partner >= 1 {
                    values[(*partner - 1) as usize] = 1.0;
                }
                if *equivalent <= 90 && *equivalent >= 1 {
                    values[(*equivalent - 1) as usize] = 1.0;
                }
                if *shadow <= 90 && *shadow >= 1 {
                    values[(*shadow - 1) as usize] = 1.0;
                }
                if *code <= 90 && *code >= 1 {
                    values[(*code - 1) as usize] = 1.0;
                }
            }
        }

        let mut res = values.to_vec();
        res.extend_from_slice(&win_mac);

        res
    }

    pub fn get_one_day_data(&mut self, day: &str) -> Vec<(Matrix, Matrix)> {
        let data = self.get_game_data(day);
        let counter = self.get_counter_data("counterpart");

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
    

    pub fn get_lotto_days_data(&mut self, days: &[&str]) -> Vec<(Matrix, Matrix)> {
        
        let mut data: Vec<(Matrix, Matrix)> = vec![];

        for day in days {
            let mut res = self.get_one_day_data(day);
            data.append(&mut res);
        }

        data
    }


    pub fn into_result(&mut self, row: &[f64]) -> Vec<i32> {
        
        let mut chuncks: Vec<Vec<f64>> = vec![];
        for r in row.chunks(90) {
            chuncks.push(r.into());
        }

        let mut res: Vec<i32> = vec![];
        for (i, v) in chuncks[0].iter().enumerate() {
            if *v >= 0.5 {
                res.push((i + 1) as i32);
            }
        }

        self.winners(&res, &chuncks[1].as_slice())
    }

    
    fn winners(&mut self, candidate: &[i32], win_mac: &[f64]) -> Vec<i32> {

        let counter = self.get_counter_data("counterpart");
 
        let mut win = vec![];
        for (_, c) in counter {
            let mut score = 0;
            let CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code} = c;
            if candidate.contains(&number) {}
            if candidate.contains(&counter) {
                score = score + 1;
            }
            if candidate.contains(&bonanza) {
                score = score + 1;
            }
            if candidate.contains(&stringkey) {
                score = score + 1;
            }
            if candidate.contains(&turning) { 
                score = score + 1;
            }
            if candidate.contains(&melta) {
                score = score + 1;
            }
            if candidate.contains(&partner) {
                score = score + 1;
            }
            if candidate.contains(&equivalent) {
                score = score + 1;
            }
            if candidate.contains(&shadow) {
                score = score + 1;
            }
            if candidate.contains(&code) {
                score = score + 1;
            }

            if score >= 5 {
                win.push(number);
            }
        }

        let mut goals = vec![];
        for w in win {
            if win_mac[(w - 1) as usize] >= 0.5 {
                goals.push(w);
            }
        }

        goals
    }

    /*
    pub fn into_inputs_bits(&mut self, row: &[i32]) -> Matrix {
        let counter = self.get_counter_data("counterpart");

        let res = self.get_row_bits(row, &counter);
        Array::from_shape_vec((res.len(), 1), res).unwrap()
    }


    pub fn get_friday_data_bits(&mut self, size: usize, day: &str) -> Vec<(Vec<Matrix>, Matrix)> {
        let data = self.get_game_data(day);
        let mut features_labels: Vec<(Vec<Matrix>, Matrix)> = Vec::new();
        let data_len = data.len() - 1;

        for n in 1..data_len  {
            // input is here
            let mut inputs: Vec<Matrix> = vec![];
            if data_len < (n + size) {
                break;
            }
            
            for i in 0..size {
                let event = data.get(&((n + i) as i32)).unwrap();
                let res: [i32; 10] = [event.w_1, event.w_2, event.w_3, event.w_4, event.w_5, event.m_1, event.m_2, event.m_3, event.m_4, event.m_5];
                let mut res_vec = vec![];
                for (i, n) in res.iter().enumerate() {
                    res_vec.push(n + ((i + 1) as i32));
                }
                let sum: i32 = res_vec.iter().sum();
                let sum_bits = self.get_int_to_class(sum);

                inputs.push(Array::from_shape_vec((sum_bits.len(), 1), sum_bits).unwrap());
            }
            
            let output_event = data.get(&((n + size) as i32)).unwrap();
            let res_vec: [i32; 10] = [output_event.w_1, output_event.w_2, output_event.w_3, output_event.w_4, output_event.w_5, output_event.m_1, output_event.m_2, output_event.m_3, output_event.m_4, output_event.m_5];
            let sum: i32 = res_vec.iter().sum();
            let sum_bits = self.get_int_to_class(sum);
            let outputs = Array::from_shape_vec((sum_bits.len(), 1), sum_bits).unwrap();
            

            features_labels.push((inputs, outputs));
        }

        features_labels
    }

    fn get_int_to_class(&self, sum: i32) -> Vec<f64> {
        let mut bits: [f64; 801] = [0.0; 801];

        for i in 0..=800 {
            if (i + 110) as i32 == sum {
                bits[i] = 1.0;
            }
        }

        bits.to_vec()
    }

    fn get_row_bits(&self, row: &[i32], counter: &HashMap<i32, CounterData>) -> Vec<f64> {
        let mut bits: Vec<f64> = vec![];

        for n in row {
            bits.extend_from_slice(&self.get_counter_part_network_by_number(n, &counter).as_slice());
        }

        bits
    }


    pub fn get_friday(&mut self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let size: usize = 1;
        let data = self.get_game_data("friday");
        let counter = self.get_counter_data("counterpart");

        let mut friday_input_vec: Vec<Vec<f64>> = Vec::new();
        let mut friday_output_vec: Vec<Vec<f64>> = Vec::new();
        let data_len = data.len() - 1;

        for n in 1..data_len  {
            if data_len < (n + size) {
                break;
            }

            // input is here
            let event = data.get(&(n as i32)).unwrap();
            friday_input_vec.push(self.get_row_coord(
                &vec![
                    event.w_1,
                    event.w_2,
                    event.w_3,
                    event.w_4,
                    event.w_5,
                    event.m_1,
                    event.m_2,
                    event.m_3,
                    event.m_4,
                    event.m_5
                ], 
                &counter
            ));

            // output is here
            let output_event = data.get(&((n + size) as i32)).unwrap();
            friday_output_vec.push(
                self.get_row_coord(
                    &vec![
                        output_event.w_1, 
                        output_event.w_2,
                        output_event.w_3,
                        output_event.w_4,
                        output_event.w_5,
                        output_event.m_1,
                        output_event.m_2,
                        output_event.m_3,
                        output_event.m_4,
                        output_event.m_5
                    ],
                    &counter
            ));
            //break;
        }

        (friday_input_vec, friday_output_vec)        
    }


    
    pub fn get_counter_part_network(&mut self) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let counter = self.get_counter_data("counterpart");

        let mut counter_input_vec: Vec<Vec<f64>> = Vec::new();
        let mut counter_output_vec: Vec<Vec<f64>> = Vec::new();
        
        for i in 1..91 {
            let CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code} = counter.get(&i).unwrap(); 
            
            let input = self.get_counter_bits(&[*number]);
            let output = self.get_counter_bits(&[*counter, *bonanza, *stringkey, *turning, *melta, *partner, *equivalent, *shadow, *code]);

            counter_input_vec.push(input.to_vec());
            counter_output_vec.push(output.to_vec());

        }

  
        (counter_input_vec, counter_output_vec)   
    }

    pub fn counter_part_to_number(&mut self, row: &[i32]) -> i32 {
        let counter = self.get_counter_data("counterpart");

        let mut res = 100;
        for (_, part) in counter.iter() {
            let CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code} = part;
            let mut score = 0;
            if *counter == row[0] {
                score = score + 1
            }
            if *bonanza == row[1] {
                score = score + 1
            }
            if *stringkey == row[2] {
                score = score + 1
            }
            if *turning == row[3] {
                score = score + 1
            }
            if *melta == row[4] {
                score = score + 1
            }
            if *partner == row[5] {
                score = score + 1
            }
            if *equivalent == row[6] {
                score = score + 1
            }
            if *shadow == row[7] {
                score = score + 1
            }
            if *code == row[8] {
                score = score + 1
            }

            if score > 5 {
                res = *number;
                break;
            }
        }
        
        res
    }

    pub fn get_counter_part_network_by_number(&self, index: &i32, counter: &HashMap<i32, CounterData>) -> Vec<f64> {
        
        let CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code} = counter.get(&index).unwrap(); 
            
        self.get_counter_bits(&[*counter, *bonanza, *stringkey, *turning, *melta, *partner, *equivalent, *shadow, *code])  
    }

    fn get_counter_bits(&self, data: &[i32]) -> Vec<f64> {

        let mut res = vec![];
        let get_bits = |d: i32| -> Vec<f64> {
            
            let mut bit_res: Vec<f64> = vec![];

            for i in format!("{:07b}", d).chars() {
                bit_res.push((i.to_string()).parse::<i32>().unwrap() as f64);
            }
            bit_res
        };

        for v in data {
            res.extend_from_slice(get_bits(*v).as_slice());
        }

        res
    }


    fn get_row_coord(&self, row: &[i32], counter_part: &HashMap<i32, CounterData>) -> Vec<f64> {
        let mut res: Vec<f64> = Vec::new();

        for n in row {
            let mut row_res: Vec<f64> = Vec::new();
            
            for (_ind, CounterData{number, counter, bonanza, stringkey, turning, melta, partner, equivalent, shadow, code}) in counter_part {
                if *counter == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.00);
                }
                if *bonanza == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.01);
                }
                if *stringkey == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.02);
                }
                if *turning == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.03);
                }
                if *melta == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.04);
                }
                if *partner == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.05);
                }
                if *equivalent == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.06);
                }
                if *shadow == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.07);
                }
                if *code == *n {
                    row_res.push(*number as f64 / 100.0);
                    row_res.push(0.08);
                }
            }

            for d in &row_res {
                res.push(*d);
            }
            
            for _ in 0..(24 - row_res.len()) {
                res.push(0.0);
            }
        }
        
        res
    }
    */
}