



pub struct _Bits;

impl _Bits {
    pub fn _bits_into_integer(bits: &[u8]) -> u8 {
        bits.iter()
            .fold(0, |result, &bit| {
                (result << 1) ^ bit
            })
    }
    
    pub fn _float_into_bits(ch: &[f64]) -> Vec<u8> {
        let mut ch_bits = vec![];
        for c in ch {
            if *c > 0.49 {
                ch_bits.push(1 as u8);
            }
            else {
                ch_bits.push(0 as u8);
            }
        }
        ch_bits
    }

    pub fn _int_into_bits(data: i32) -> Vec<f64> {
        let mut bit_res: Vec<f64> = vec![];

        for i in format!("{:07b}", data).chars() {
            bit_res.push((i.to_string()).parse::<i32>().unwrap() as f64);
        }

        bit_res
    }
}


