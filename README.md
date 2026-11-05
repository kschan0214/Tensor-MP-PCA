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

`denoise_mppca_batched.m` is a vectorised implementation of the **matrix case** of `denoise_recursive_tensor`: 4D data `[x y z Nmeas]` with the default options (indices `{1:3, 4}`, optimal shrinkage, stride 1, overlapping patches averaged). The original loops over patches with one SVD each; this version stacks the patches as pages and decomposes them with MATLAB's `pagesvd`, which is multithreaded across pages. The algorithm is unchanged:
- Marchenko–Pastur noise estimate;
- component cut-off;
- optimal Frobenius shrinkage;
- patch averaging.

```matlab
[denoised, Sigma2, P, SNR_gain, info] = denoise_mppca_batched(data, [5 5 5], 'precision', 'double');
% options: 'mask' (skip windows with no mask voxel), 'batch' (patches per batch, default 20000),
%          'precision' ('double' default, or 'single')
```

**Validation** against `denoise_recursive_tensor(data,[5 5 5])`:
- in vivo diffusion MRI, real-valued (phase-corrected) DWI at 3T (Cima.X);
- 12-slice slab, 10 acquisitions × 2 echo times (5–105 volumes each);
- MATLAB R2024b, 40 threads.

| Version | Time | Max difference | σ and SNR gain |
| --- | --- | --- | --- |
| `denoise_recursive_tensor` | 1112 s | reference | reference |
| `denoise_mppca_batched`, double | 55 s (20×) | 1.9e-13 σ | identical to machine precision |
| `denoise_mppca_batched`, single | 35 s (32×) | 0.02 σ (RMS 3e-5 σ) | max relative σ difference 7e-5 |

A full volume of 110×110×75×105 takes about 70 s in single precision (about 40 s with a head mask).

**Limitations:**
- Only the matrix case is batched; tensors with three or more indices still use `denoise_recursive_tensor`.
- CPU only: MATLAB's `gpuArray` `pagesvd` accepts at most 32×32 pages.

Please cite Olesen et al., Magn Reson Med 2022 (doi:10.1002/mrm.29478) when using either function.
