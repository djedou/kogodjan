
/// Trait describing nodes that can accept new values once
/// the graph has been defined.
pub trait DataInput<T> {
    /// Set the value of this node.
    fn set_value(&self, input: T);
}