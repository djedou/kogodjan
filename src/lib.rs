//pub extern crate nalgebra;
//pub extern crate rand;
//#[macro_use]
//extern crate serde_derive;
#[macro_use] 
extern crate random_number;

pub mod layers;
pub mod neural_traits;
pub mod network_arch;
pub mod activators;
pub mod loss_functions;
pub mod maths;
//pub mod maths;
pub mod optimizers;
pub mod utils;
pub use djed_maths;