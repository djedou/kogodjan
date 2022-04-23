use crate::maths::Matrix;
use std::cell::{Cell, RefCell};
use serde_derive::{Serialize, Deserialize};

/// Struct used to hold parameters that need to be shared among
/// multiple `ParameterNode`s for asynchronous, parallel optimization.
#[derive(Debug, Serialize, Deserialize)]
pub struct HogwildParameter {
    shape: (usize, usize),
    pub(crate) value: RefCell<Matrix>,
    pub(crate) squared_gradients: RefCell<Matrix>,
    pub(crate) moments: RefCell<Matrix>,
    num_updates: Cell<i32>,
}

impl Clone for HogwildParameter {
    fn clone(&self) -> HogwildParameter {
        HogwildParameter {
            shape: self.shape.clone(),
            value: self.value.clone(),
            squared_gradients: self.squared_gradients.clone(),
            moments: self.moments.clone(),
            num_updates: self.num_updates.clone(),
        }
    }
}


impl HogwildParameter {
    /// Create a new parameter object.
    pub fn new(value: Matrix) -> Self {
        let squared_gradients = &value * 0.0;
        let moments = &value * 0.0;
        let shape = (value.nrows(), value.ncols());

        HogwildParameter {
            shape: shape,
            value: RefCell::new(value),
            squared_gradients: RefCell::new(squared_gradients),
            moments: RefCell::new(moments),
            num_updates: Cell::new(0),
        }
    }

    /// Get the parameter value.
    pub fn value(&self) -> &Matrix {
        unsafe { &*(self.value.as_ptr()) }
    }

    pub(crate) unsafe fn value_mut(&self) -> &mut Matrix {
        &mut *(self.value.as_ptr())
    }

    pub(crate) unsafe fn squared_gradient_mut(&self) -> &mut Matrix {
        &mut *(self.squared_gradients.as_ptr())
    }

    pub(crate) unsafe fn moments_mut(&self) -> &mut Matrix {
        &mut *(self.moments.as_ptr())
    }

    pub(crate) unsafe fn num_updates_mut(&self) -> &mut i32 {
        &mut *(self.num_updates.as_ptr())
    }
}


unsafe impl Sync for HogwildParameter {}
