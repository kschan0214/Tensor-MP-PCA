# Changes in this fork

Fork of [Neurophysics-CFIN/Tensor-MP-PCA](https://github.com/Neurophysics-CFIN/Tensor-MP-PCA) (upstream commit `60c8b86`), GPL-3.0.

## 2026-11 — Kwok-Shing Chan (MGH)

- Added `denoise_mppca_batched.m`: a batched (`pagesvd`) implementation of the matrix case of `denoise_recursive_tensor`. It gives the same results (double precision, max difference 1.9e-13 σ) and runs 20–32× faster; validation is in README.md.
- README: added the "Batched implementation (fork addition)" section.
- The original files (`denoise_recursive_tensor.m`, `denoise_array_recursive_tensor.m`, `opt_shrink_frob.m`, `LICENSE`) are unchanged.
