# Changes in this fork

Fork of [Neurophysics-CFIN/Tensor-MP-PCA](https://github.com/Neurophysics-CFIN/Tensor-MP-PCA) (upstream commit `60c8b86`), GPL-3.0.

## 2026-11 — Kwok-Shing Chan (MGH)

- Added `denoise_mppca_batched.m`: a batched (`pagesvd`) implementation of the matrix case of `denoise_recursive_tensor`. It gives the same results (double precision, max difference 1.9e-13 σ) and runs 20–32× faster; validation is in README.md.
- README: added the "Batched implementation (fork addition)" section.
- The original files (`denoise_recursive_tensor.m`, `denoise_array_recursive_tensor.m`, `opt_shrink_frob.m`, `LICENSE`) are unchanged.

## 2026-11-05 — Kwok-Shing Chan (MGH)

- Added `denoise_recursive_tensor_batched.m`: a batched implementation of the full `denoise_recursive_tensor` (all options, matrix and tensor cases). It matches the original to rounding error (17 option cases; real DWI and multi-echo GRE data) and runs about 20–30× faster.
- `subtract_mean` falls back to the original.
- The SNR_gain formula is extended for `num_inds` smaller than the number of indices (the original errors in that case).
- README: the batched section now describes the general function.
