mod data_iter;
mod synthetic_data;
mod extern_gradient;
//mod network_parameters;

pub(crate) use data_iter::*;
pub use synthetic_data::*;
pub use extern_gradient::*;
//pub(crate) use network_parameters::*;

/*
pub fn round(value: f64) -> f64 {
    (value * 100000.0).round() / 100000.0
}*/