use nalgebra::{Matrix, Dynamic, VecStorage};


/// `MatrixD` is a 2-D array of numbers  
/// docs: [`nalgebra::Matrix`](https://nalgebra.org/vectors_and_matrices/#the-generic-matrix-type) and [` nalgebra::base::Matrix`](https://nalgebra.org/rustdoc/nalgebra/base/struct.Matrix.html)
pub type MatrixD<A> = Matrix<A, Dynamic, Dynamic, VecStorage<A, Dynamic, Dynamic>>;

/*
/// We can think of Vectors as identifying points in space, with each element giving   
/// the coordinate along a different axis.  
/// `Vector` is a `Matrix` of `n rows and 1 column` of numbers and uses all methos on [`ndarray::ArrayBase`](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html)   
pub type Vector<A> = Array2<A>;

/// `Tensor` is a 3-D array of numbers and uses all methos on [`ndarray::ArrayBase`](https://docs.rs/ndarray/0.13.0/ndarray/struct.ArrayBase.html)   
pub type Tensor<A> = Array3<A>;*/

