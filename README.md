# Additive FFT Reed-Solomon Codec

This library contains an implementation of the fastest to date Reed-Solomon encode and decode algorithms presented in [LNH][lnh] and [LCH][lch].

In addition to the scalar encoder and the error-correcting decoder presented in [LNH][lnh], a sharded encoder and a sharded erasure-only decoder are designed and implemented here.

An (n,k) RS code appends T=n-k parity symbols to the k message symbols, forming a codeword of length n. (n,k) RS codes can correct up to floor(T/2) erroneous symbols (shards) when the locations of those errors are not known, and up to T erasures (shards) when erasure locations are known.


## On additive RS

Additive RS codes is one of the two classes of RS codes, with the other class being the multiplicative RS codes.

❗⚡ **Additive RS codes are not wire-compatible with multiplicative RS codes.** ⚡❗

Multiplicative RS codes are the most common due to posessing a textbook implementation. Libraries such as [reed-solomon-erasure][rse], [ISA-L][isa-l], [Backblaze][backblaze] and [Klaus Post's reedsolomon][klauspost] all implement multiplicative RS codes. In addition, all four are erasure-only - broken shard positions must be known in advance. The multiplicative RS implementations evaluate the message polynomial at elements of the multiplicative group of GF(2^8), a cyclic group of order 255. This limits the code length to 255 and makes encoding an O(k·n) product of a precomputed Cauchy generator matrix and the message vector. Decoding is O(k^3), which comes from Gaussian elimination of the k×k submatrix of the encoding matrix corresponding to the correct rows. Some implementations amortise this by precomputing or caching inverses for common erasure patterns, but the one-time cost is still O(k^3).

Additive RS codes evaluate the message polynomial at all 256 elements of the additive group, a GF(2)-vector space, whose power subspace structure admits a radix-2 FFT. Encoding and decoding both run in O(n log(n)) field operations, and the natural code length is 256.

||ISA-L, reed-solomon-erasure, Backblaze|Cauchy with structure|LNH additive FFT|
|---|---|---|---|
|encoding         |O(k·T) |O(k·T) |O(n log(T))                 |
|erasure decoding |O(k^3) |O(k^2) |O(n log(T))                 |
|error correction | ❌    |O(n·T) |O(n log(T) + T log(log(T))) |


## Benchmarks

Results are dependent on hardware, so you are advised to run `cargo bench`.

Here is a representative set of results, time and throughput, obtained on an AMD EPYC 9575F for n=64 and k=32.

|shard length, bytes|encode 32 message shards|recover 32 random shard erasures|
|---|---|---|
| 64 | 385ns, 10GiB/s | 7.7µs, 510MiB/s |
| 1k | 2.4µs, 25GiB/s | 14.µs, 4.1GiB/s |
| 64k | 192µs, 20GiB/s | 740µs, 5.3GiB/s |


## Possible usecases

### Best fit

- Large n, large shards, e.g. n ≥ 128, hundreds of kilobytes or megabytes.

- Medium redundancy ratio, k = n/2. The additive FFT cost is expressed in terms of n regardless of how it splits into k and T. Cauchy decoding, on the other hand, is faster when k is low and the redundancy ratio is high.

- Erasure-heavy environments - distributed storage, high throughput erasure-coded gossip messaging.

- x86-64 with AVX-512 GFNI. The GFNI FFT butterfly gives the largest absolute throughput.

### Not a good fit

- Small n, e.g. n ≤ 16. The FFT setup overhead can dominate.

- Small shards - tens of bytes. Per-butterfly ovehead dominates.

- Wire compatibility required. The crate is not Cauchy implementation compatible.

- Non-x86 accelerated targets. We have no support for ARM NEON or RISC-V RVV acceleration yet.

- Correction of quietly corrupted shards. Sharded error correction is not implemented yet.

- n > 256. GF(2^8) caps the codeword length at 256.


## References

- Sian-Jheng Lin, Tareq Y. Al-Naffouri, Yunghsiang S. Han.
  [*FFT Algorithm for Binary Extension Finite Fields and its Application to Reed-Solomon Codes*][lnh].
  arXiv:1503.05761, 2016.

- Sian-Jheng Lin, Wei-Ho Chung, Yunghsiang S. Han.
  [*Novel Polynomial Basis with Fast Fourier Transform and Its Application to Reed-Solomon Erasure Codes*][lch].
  FOCS 2014.

[lnh]: https://arxiv.org/abs/1503.05761
[lch]: https://arxiv.org/abs/1404.3458
[rse]:      https://github.com/rust-rse/reed-solomon-erasure
[isa-l]:    https://github.com/intel/isa-l
[backblaze]: https://github.com/Backblaze/JavaReedSolomon
[klauspost]: https://github.com/klauspost/reedsolomon
