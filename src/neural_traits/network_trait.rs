use crate::errors::NetworkErrorFunction;

pub trait NetworkT {

    fn train(&mut self, rate: f64, error_func: NetworkErrorFunction);
}