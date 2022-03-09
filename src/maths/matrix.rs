use ndarray::{Array2, arr2, FixedInitializer};


#[derive(Debug, Clone)]
pub struct Matrix<A>
where 
    A: Clone + Copy
{
    nrows: usize,
    ncols: usize,
    data: Array2<A>
}


impl<A> Matrix<A> 
where 
    A: Clone + Copy
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

    pub fn new_from_array2(xs: &Array2<A>) -> Matrix<A> {
        Matrix {
            nrows: xs.nrows(),
            ncols: xs.ncols(),
            data: xs.clone()
        }
    }

    pub fn get_nrows(&self) -> usize {
        self.nrows
    }
    
    pub fn get_ncols(&self) -> usize {
        self.ncols
    }

    pub fn get_data(&self) -> Array2<A> {
        self.data.clone()
    }
}