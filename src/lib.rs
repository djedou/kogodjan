pub extern crate nalgebra;
pub extern crate rand;
#[macro_use]
extern crate serde_derive;

pub mod layers;
pub mod neural_traits;
pub mod network_arch;
pub mod activators;
pub mod loss_functions;
pub mod maths;
pub mod optimizers;
//pub(crate) mod utils;
pub mod utils;