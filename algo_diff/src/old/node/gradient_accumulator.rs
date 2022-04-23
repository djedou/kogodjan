use crate::maths::{Matrix, ArraySliceMut, ArraySliceOps};
use hibitset::BitSet;
use itertools::izip;
use crate::{clamp};
use ndarray::Axis;
use std::cell::Ref;


#[derive(Debug)]
pub(crate) struct GradientAccumulator {
    gradient: Matrix,
    sparse_index: BitSet,
    dense: bool,
    sparse: bool,
}

impl GradientAccumulator {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            gradient: Matrix::zeros(shape),
            sparse_index: BitSet::with_capacity(100),
            dense: false,
            sparse: false,
        }
    }

    pub fn add_dense(&mut self, grad: &Matrix) {
        if !self.dense {
            self.gradient.slice_assign(grad);
        } else {
            self.gradient.slice_add_assign(grad);
        }

        self.dense = true;
    }

    pub fn add_sparse(&mut self, indices: &[usize], grad: &Matrix) {
        for (&idx, row) in izip!(indices.iter(), grad.rows().into_iter()) {
            self.add_sparse_row(idx, &row);
        }
    }

    pub fn add_sparse_row(&mut self, idx: usize, grad: &ndarray::ArrayView<f64, ndarray::Ix1>) {
        if self.sparse_index.add(idx as u32) {
            self.gradient
                .index_axis_mut(Axis(0), idx)
                .slice_add_assign(grad);
        } else {
            self.gradient.index_axis_mut(Axis(0), idx).slice_assign(grad);
        }

        self.sparse = true;
    }

    pub fn sparse_iter(
        &self,
    ) -> impl Iterator<Item = (usize, ndarray::ArrayView<f64, ndarray::Ix1>)> {
        let idx = &self.sparse_index;
        let grad = &self.gradient;

        idx.into_iter().map(move |idx| {
            let idx = idx as usize;
            (idx, grad.index_axis(Axis(0), idx))
        })
    }

    pub fn zero_gradient(&mut self) {
        if self.sparse {
            self.sparse_index.clear()
        }

        self.dense = false;
        self.sparse = false;
    }

    pub fn gradient(&self) -> &Matrix {
        &self.gradient
    }

    /// With sparse gradients we don't reset to zero, so we
    /// need this to provide correct dense gradients to
    /// finite difference methods.
    pub fn materialized_gradient(&self) -> Matrix {
        if self.has_dense() {
            self.gradient.clone()
        } else {
            let mut grad = &self.gradient * 0.0;
            for (idx, row) in self.sparse_iter() {
                grad.index_axis_mut(Axis(0), idx).slice_assign(&row);
            }
            grad
        }
    }

    pub fn has_dense(&self) -> bool {
        self.dense
    }

    pub fn clamp(&mut self, min: f64, max: f64) {
        if self.has_dense() {
            self.gradient
                .fast_slice_mut()
                .iter_mut()
                .for_each(|x| *x = clamp(*x, min, max));
        } else {
            unimplemented!();
            // for (idx, row) in self.sparse_iter() {
            //     self.gradient
            //         .subview_mut(Axis(0), idx)
            //         .fast_slice_mut()
            //         .iter_mut()
            //         .for_each(|x| *x = clamp(*x, min, max));
            // }
        }
    }
}



pub trait GradientSink<T> {
    fn accumulate_gradient(&mut self, gradient: T);
}

impl<'a, 'b> GradientSink<&'a Ref<'b, Matrix>> for GradientAccumulator {
    fn accumulate_gradient(&mut self, gradient: &Ref<Matrix>) {
        self.add_dense(gradient);
    }
}

impl<'a> GradientSink<&'a Matrix> for GradientAccumulator {
    fn accumulate_gradient(&mut self, gradient: &'a Matrix) {
        self.add_dense(gradient);
    }
}

impl<'a> GradientSink<&'a mut Matrix> for GradientAccumulator {
    fn accumulate_gradient(&mut self, gradient: &'a mut Matrix) {
        self.add_dense(gradient);
    }
}

impl<'a> GradientSink<(&'a [usize], &'a Matrix)> for GradientAccumulator {
    fn accumulate_gradient(&mut self, gradient: (&'a [usize], &'a Matrix)) {
        let (idx, grad) = gradient;
        self.add_sparse(idx, grad);
    }
}

impl<'a, 'b: 'a> GradientSink<(usize, &'a ndarray::ArrayView<'b, f64, ndarray::Ix1>)>
    for GradientAccumulator
{
    fn accumulate_gradient(
        &mut self,
        gradient: (usize, &'a ndarray::ArrayView<'b, f64, ndarray::Ix1>),
    ) {
        let (idx, grad) = gradient;
        self.add_sparse_row(idx, grad);
    }
}
