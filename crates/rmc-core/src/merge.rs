/// Combine independent partial results.
///
/// `Merge` is the reduction contract used by parallel chains. Implementations must represent a
/// quantity that is valid to combine directly. For scalar numeric impls below this means "sum-like
/// totals" (counts, accumulated sums, log weights, etc.), not already-normalized per-chain means.
/// If each chain computes a mean, wrap the numerator/count in a dedicated accumulator type and
/// merge those fields instead.
pub trait Merge {
    fn merge(self, other: Self) -> Self;
}

impl Merge for () {
    fn merge(self, _other: Self) -> Self {}
}

impl Merge for u64 {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}

impl Merge for usize {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}

impl Merge for i64 {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}

impl Merge for f64 {
    fn merge(self, other: Self) -> Self {
        self + other
    }
}

macro_rules! impl_merge_tuple {
    ($($t:ident $idx:tt),+) => {
        impl<$($t: Merge),+> Merge for ($($t,)+) {
            fn merge(self, other: Self) -> Self {
                ($(self.$idx.merge(other.$idx),)+)
            }
        }
    };
}

impl_merge_tuple!(T0 0, T1 1);
impl_merge_tuple!(T0 0, T1 1, T2 2);
impl_merge_tuple!(T0 0, T1 1, T2 2, T3 3);
