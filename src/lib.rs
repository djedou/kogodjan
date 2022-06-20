mod activator;
mod loss;
mod utils;

pub mod layers;
pub mod networks;
pub mod io;
pub use algo_diff::{
    maths::{Matrix, Array, Axis}
};
pub use loss::*;
pub use activator::*;
