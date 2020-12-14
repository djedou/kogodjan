/*use crate::maths::types::MatrixD;

//ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>>
pub trait MatrixT<D> {
    fn mat_mul(&self, other: &MatrixD<D>);
}


impl<D> MatrixT<D> for MatrixD<D>
where D: std::ops::Mul<Output = D>
{
    fn mat_mul(&self, other: &MatrixD<D>){
        let mut sr = 0;
        let mut sc = 0;
        let mut or = 0;
        let mut oc = 0;

        if let &[r, c] = self.shape() {
            sr = r;
            sc = c;
        }
        if let &[r, c] = other.shape() {
            or = r;
            oc = c;
        }

        match sc == or {
            true => {
                let mul = Array2::from_shape_fn((sr, oc), |(a, b)| {
                    
                    println!("a: {:?} b: {:?}", a, b);
                    //println!("self row : {:?}",a )
                    //self.row(a).dot(other.column(b).reversed_axes())
                });

                //0.0
            },
            false => {
                panic!("columns of self should equal to rows other!");
            }
        }
    }
}


#[cfg(test)]
mod matrix_trait_test {
    use ndarray::{Array, array, Array2};
    use rand::{thread_rng,Rng};
    use crate::maths::types::{Vector, Matrix};
    use crate::maths::algebra::MatrixT;
    #[test]
    fn mut_mul_test() {
        let true_w: Vector<f64> = array![[2.0], [-3.4]];

        let mut rng = thread_rng(); 
    
        let x: Array2<f64> = Array::from_shape_fn((5, 2), |_args| {
                
            let value = rng.gen::<f64>(); // generate float between 0.0 and 1.0
            value
        });

        x.mut_mal(&true_w);
    }
}
*/