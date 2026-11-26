/// The `R * d` shards of a radix-R FFT node starting at shard `base`.
pub struct Node<'a, G, const R: usize> {
    shards: &'a mut [G],
    shard_len: usize,
    base: usize,
    d: usize,
}

pub type Dit2Node<'a, G> = Node<'a, G, 2>;
pub type Dit4Node<'a, G> = Node<'a, G, 4>;

/// R-tuple shards of equal length by construction, ensured by `Node::split`.
pub(crate) struct ShardTuple<'a, G, const R: usize>([&'a mut [G]; R]);

impl<'a, G, const R: usize> Node<'a, G, R> {
    /// The `R * d` shards of an FFT node starting at shard `base`.
    pub(crate) fn split(self) -> impl Iterator<Item = ShardTuple<'a, G, R>> {
        let q = self.d * self.shard_len;
        let node = &mut self.shards[self.base * self.shard_len..][..R * q];
        let mut parts = node
            .chunks_exact_mut(q)
            .map(|p| p.chunks_exact_mut(self.shard_len));
        let mut parts: [_; R] = std::array::from_fn(|_| parts.next().unwrap());
        (0..self.d).map(move |_| ShardTuple(parts.each_mut().map(|p| p.next().unwrap())))
    }
}

impl<'a, G, const R: usize> ShardTuple<'a, G, R> {
    #[inline]
    pub(crate) fn into_shards(self) -> [&'a mut [G]; R] {
        self.0
    }
}

impl<'a, G> ShardTuple<'a, G, 2> {
    /// Splits two equal-length shards into lanes of W-element chunks and the remaining tails.
    #[inline]
    pub(crate) fn into_chunks<const W: usize>(
        self,
    ) -> (impl Iterator<Item = [&'a mut [G; W]; 2]>, [&'a mut [G]; 2]) {
        let [s0, s1] = self.0;
        let (c0, t0) = s0.as_chunks_mut::<W>();
        let (c1, t1) = s1.as_chunks_mut::<W>();
        (c0.iter_mut().zip(c1).map(|(p0, p1)| [p0, p1]), [t0, t1])
    }
}

impl<'a, G> ShardTuple<'a, G, 4> {
    /// Splits four equal-length shards into lanes of W-element chunks and the remaining
    /// tails.
    #[inline]
    pub(crate) fn into_chunks<const W: usize>(
        self,
    ) -> (impl Iterator<Item = [&'a mut [G; W]; 4]>, [&'a mut [G]; 4]) {
        let [s0, s1, s2, s3] = self.0;
        let (c0, t0) = s0.as_chunks_mut::<W>();
        let (c1, t1) = s1.as_chunks_mut::<W>();
        let (c2, t2) = s2.as_chunks_mut::<W>();
        let (c3, t3) = s3.as_chunks_mut::<W>();
        let cs = c0
            .iter_mut()
            .zip(c1)
            .zip(c2)
            .zip(c3)
            .map(|(((p0, p1), p2), p3)| [p0, p1, p2, p3]);
        (cs, [t0, t1, t2, t3])
    }
}
