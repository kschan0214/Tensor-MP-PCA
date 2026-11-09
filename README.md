# Tensor-MP-PCA
Matlab implementation of MP-PCA based denoising of multidimensional data based on Olesen, JL, Ianus, A, Østergaard, L, Shemesh, N, Jespersen, SN. Tensor denoising of multidimensional MRI data. Magn Reson Med. 2022; 1- 13. doi:10.1002/mrm.29478. Usage is free, but please cite the paper above. The data can also be shared, but due to file size (~40 Gb), must be arranged via file transfer by contacting sune@cfin.au.dk.

Syntax 

[denoised,Sigma2,P,SNR_gain] = denoise_recursive_tensor(data,window,varargin)

MP-PCA based denoising of multidimensional (tensor structured) data.

Input variables. 
data: data with noise window: window to pass over data with. Typically
voxels. 

window = [5 5] would process patches of data with dimensions 5x5x...

varargin: is specified as name-value pairs (i.e. ...,'mask',mask,...)
             indices: determines tensor-struture of patches. For instance
             for data with 5 indices with the three first being voxels,
             indices = {1:3 4 5} will combine the voxels into one index
             and retain the others so that each patch is denoised as a
             three-index tensor -- indices = {1:2 3 4 5} would denoise
             each patch as a four-index tensor and so on. It defaults to
             combining the voxel/window indices and sorting according to
             index dimensionality in ascending order, since this appears
             to be optimal in most cases. mask: if a logical mask is
             specified, locations were the sliding window contains no
             "true" voxels are skipped. opt_shrink: uses optimal shrinkage
             if true (default is false) sigma: specifies a know value for
             the noise sigma rather than estimating it using
             MP-distribution

Output: 
 
denoised: the denoised data 
 
Sigma2: estimated noise variance 
 
P: estimated number signal components 
 
SNR_gain: an estimate of the estimated gain in signal-to-noise ratio.

## Batched implementation (fork addition)

`denoise_recursive_tensor_batched.m` is a batched (vectorised) version of `denoise_recursive_tensor` with **the same inputs, options and outputs**. The original loops over sliding-window patches with one SVD each. This version processes all patches of a batch together, as pages of MATLAB's `pagesvd`, which is multithreaded across pages.

In the tensor case the number of retained components P differs between patches. Patches are therefore zero-padded to full rank, and each patch's true matrix sizes are tracked, so the Marchenko–Pastur cut-offs, combined noise estimate and optimal shrinkage are exactly those of the original.

```matlab
[denoised, Sigma2, P, SNR_gain, info] = denoise_recursive_tensor_batched(data, window, varargin)
% same options as denoise_recursive_tensor: indices, mask, stride, center_assign, opt_shrink,
% sigma2 (known noise), full_sigma2_pass, num_inds
% extra options: 'batch' (patches per batch, default 20000), 'precision' ('double' default | 'single')
```

**Parity with the original:** every option was compared on synthetic low-rank data, 17 cases including 4D and 5D data, all `indices` groupings, mask, center_assign, stride, known sigma, opt_shrink off, full_sigma2_pass off and num_inds. The denoised data agree within 3e-13 of the noise SD; P and Sigma2 are identical to rounding.

**Real data (MATLAB R2024b, 40 threads):**

| Data | Case | Original | Batched | Max difference |
| --- | --- | --- | --- | --- |
| DWI, real-valued, 3T, 12 slices, 10 acquisitions × 2 TE | matrix `[vox × volumes]` | 1026 s | 54 s (19×) | 1.9e-13 σ |
| multi-echo GRE (15 echoes × 3 flip angles), 12 slices, real part | tensor `{vox, echo, FA}` | 187 s | 7 s (27×) | 1.4e-13 σ |

**Notes:**
- `'subtract_mean', true` is not batched; it falls back to `denoise_recursive_tensor`, with a warning.
- With `'num_inds'` smaller than the number of indices, the original's `SNR_gain` line errors because P has fewer columns than the patch size vector. The batched version counts the indices it does not denoise as full rank in that formula.
- It is CPU only: MATLAB's `gpuArray` `pagesvd` accepts at most 32×32 pages.
- `denoise_mppca_batched.m` is the earlier matrix-only version, kept for compatibility.

Please cite Olesen et al., Magn Reson Med 2022 (doi:10.1002/mrm.29478) when using these functions.
