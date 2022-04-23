 #[derive(Debug, Clone, Copy)]
pub struct LottoData {
    pub event: i32,
    pub w_1: i32,
    pub w_2: i32,
    pub w_3: i32,
    pub w_4: i32,
    pub w_5: i32,
    pub m_1: i32,
    pub m_2: i32,
    pub m_3: i32,
    pub m_4: i32,
    pub m_5: i32,
}

#[derive(Debug, Clone)]
pub struct CounterData {
    pub number: i32,
    pub counter: i32,
    pub bonanza: i32,
    pub stringkey: i32,
    pub turning: i32,
    pub melta: i32,
    pub partner: i32,
    pub equivalent: i32,
    pub shadow: i32,
    pub code: i32
}
