use ndarray::{Array2, arr2, FixedInitializer};


#[derive(Debug)]
pub struct Matrix<A>
where 
    A: Clone
{
    pub nrows: usize,
    pub ncols: usize,
    pub data: Array2<A>
}


impl<A> Matrix<A> 
where 
    A: Clone
{
    pub fn new<V: FixedInitializer<Elem = A>>(xs: &[V]) -> Matrix<A>
    where
        V: Clone,
    {
        let arr = arr2(xs);
        Matrix {
            nrows: arr.nrows(),
            ncols: arr.ncols(),
            data: arr
        }
    }
}