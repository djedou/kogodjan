use postgres::{
    Client, 
    NoTls,
    row::Row,
    Error
};
use crate::data::{CounterData, LottoData};
use std::collections::{HashMap};

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

    pub fn _get_counter_data(&mut self, counter: &str) -> HashMap<i32 ,CounterData> {
    
        let query = format!("SELECT * FROM lotto.{:?}", counter);
    
        let mut data: HashMap<i32 ,CounterData> = HashMap::new();
    
        if let Ok(ref mut cl) = self.client {
            for rows in cl.query(&query[..], &[]) {
                rows.iter().for_each(|r| {
                    let item = self._get_counter_row(r);
                    data.insert(item.number, item);
                });
            }
        }
        
        data
    }

    fn _get_counter_row(&self, row: &Row) -> CounterData {
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
}