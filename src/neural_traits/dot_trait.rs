

/*impl<A, S, S2> Dot<ArrayBase<S2, Ix2>> for ArrayBase<S, Ix2>
where
    S: Data<Elem = A>,
    S2: Data<Elem = A>,
    A: LinalgScalar,
{
    type Output = Array2<A>;
    fn dot(&self, b: &ArrayBase<S2, Ix2>) -> Array2<A> {
        let a = self.view();
        let b = b.view();
        let ((m, k), (k2, n)) = (a.dim(), b.dim());
        if k != k2 || m.checked_mul(n).is_none() {
            dot_shape_error(m, k, k2, n);
        }

        let lhs_s0 = a.strides()[0];
        let rhs_s0 = b.strides()[0];
        let column_major = lhs_s0 == 1 && rhs_s0 == 1;
        // A is Copy so this is safe
        let mut v = Vec::with_capacity(m * n);
        let mut c;
        unsafe {
            v.set_len(m * n);
            c = Array::from_shape_vec_unchecked((m, n).set_f(column_major), v);
        }
        mat_mul_impl(A::one(), &a, &b, A::zero(), &mut c.view_mut());
        c
    }
}
*/

pub trait Dot {

    type Output;
    type Other;

    fn dot(&self, other: &Self::Other) -> Self::Output ;

}