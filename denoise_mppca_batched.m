function [denoised, Sigma2, P, SNR_gain, info] = denoise_mppca_batched(data, window, varargin)
%DENOISE_MPPCA_BATCHED  Batched (vectorised) MP-PCA, matrix case, reproducing Tensor-MP-PCA's
%   denoise_recursive_tensor(data, window) for 4D data [x y z Nmeas] with the default options
%   (indices = {1:3, 4}, opt_shrink = true, no mean subtraction, stride 1, every window
%   position fully inside the volume, overlapping patches averaged).
%
%   [denoised, Sigma2, P, SNR_gain, info] = denoise_mppca_batched(data, [5 5 5], 'mask', m, 'batch', 20000, 'precision', 'double')
%
%   Instead of one svd per patch in a serial loop, patches are stacked as pages and
%   decomposed with pagesvd (multithreaded across pages). Per patch X (prod(window) x Nmeas):
%     [U,S,V] = svd(X,'econ'); MP noise estimate -> sigma2 (estimate_noise);
%     keep components with s^2 > sigma2*(sqrt(M)+sqrt(N))^2 (discard_noise_components);
%     optimal Frobenius shrinkage s'^2 = s^2 - 2(N'+M')sigma2 + (N'-M')^2 sigma2^2/s^2,
%       M' = max(M-P,1), N' = max(N-P,1) (apply_optimal_shrinkage / opt_shrink_frob);
%     Xhat = U_P S'_P V_P'.
%   Note: MATLAB's gpuArray pagesvd is limited to 32x32 pages, so this runs on the CPU.
%
%   Options: 'mask' (logical [x y z]; windows with no true voxel are skipped, as in the CPU code),
%            'batch' (patches per batch, default 20000), 'precision' ('double' default, or 'single').

%
%   Please cite: Olesen JL, Ianus A, Østergaard L, Shemesh N, Jespersen SN. Tensor denoising
%   of multidimensional MRI data. Magn Reson Med. 2022. doi:10.1002/mrm.29478
%
%   Derived from denoise_recursive_tensor.m / denoise_array_recursive_tensor.m of
%   Tensor-MP-PCA (https://github.com/Neurophysics-CFIN/Tensor-MP-PCA, Olesen et al.).
%   Modified by Kwok-Shing Chan (MGH), November 2026: batched pagesvd implementation of the
%   matrix case (validation numbers in README.md).
%
%   This program is free software: you can redistribute it and/or modify it under the terms
%   of the GNU General Public License as published by the Free Software Foundation, either
%   version 3 of the License, or (at your option) any later version. This program is
%   distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
%   the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
%   GNU General Public License (LICENSE) for more details.

opt.mask = []; opt.batch = 20000; opt.precision = 'double';
for k = 1:2:numel(varargin), opt.(varargin{k}) = varargin{k+1}; end

t0 = tic;
dims  = size(data); if numel(dims) < 4, dims(4) = 1; end
dvox  = dims(1:3); nm = dims(4); nvox = prod(dvox);
window = reshape(window,1,[]); M = prod(window); N = nm;
if isempty(opt.mask), opt.mask = true(dvox); end

% corner positions where the window fits (same as the CPU loop)
[cx,cy,cz] = ndgrid(1:dvox(1)-window(1)+1, 1:dvox(2)-window(2)+1, 1:dvox(3)-window(3)+1);
corners = sub2ind(dvox, cx(:), cy(:), cz(:));
[wx,wy,wz] = ndgrid(0:window(1)-1, 0:window(2)-1, 0:window(3)-1);
inc = (wx(:) + wy(:)*dvox(1) + wz(:)*dvox(1)*dvox(2))';          % 1 x M offsets (ind2sub(window,...) order)
keep = any(opt.mask(corners + inc), 2);                          % skip windows with no mask voxel
corners = corners(keep);
np = numel(corners);

cls  = opt.precision;
D    = cast(reshape(data, nvox, nm), cls);
acc  = zeros(nvox, nm, cls);
cnt  = zeros(nvox, 1, cls);
s2a  = zeros(nvox, 1, cls);
pa   = zeros(nvox, 1, cls);
K    = min(M,N);
Mest = K; Nest = M*N/K;                     % estimate_noise: M = size(S,1) = min dim, N = prod/M
cfac = (sqrt(M)+sqrt(N))^2;                 % MP upper edge factor (symmetric in M,N)
kk   = cast((0:K-1)', cls);                 % candidate numbers of signal components
lin  = (0:K-1)*(K+1) + 1;                   % diagonal positions within a K x K page

for b0 = 1:opt.batch:np
    c  = corners(b0:min(np, b0+opt.batch-1));
    nb = numel(c);
    idx = c + inc;                          % nb x M voxel indices
    X  = permute(reshape(D(idx(:), :), nb, M, N), [2 3 1]);   % M x N x nb
    [U,S,V] = pagesvd(X, 'econ');
    S2 = reshape(S, K*K, nb); s2 = S2(lin, :).^2;             % K x nb, descending
    % ---- estimate_noise ----
    sig_est = flipud(cumsum(flipud(s2),1)) ./ ((Mest - kk) .* (Nest - kk));
    below   = s2 < sig_est * cfac;
    [hasP, first] = max(below, [], 1);
    sigma2 = sig_est(sub2ind([K nb], first, 1:nb));
    sigma2(~hasP) = 0;
    % ---- discard_noise_components ----
    keepc = s2 > sigma2 * cfac;
    Pk = sum(keepc, 1);
    % ---- optimal shrinkage ----
    Mp = max(M - Pk, 1); Np = max(N - Pk, 1);
    s2s = s2 - 2*(Np + Mp).*sigma2 + (Np - Mp).^2 .* sigma2.^2 ./ s2;
    ss  = real(sqrt(complex(s2s))) .* keepc;
    % ---- reconstruct and accumulate ----
    Xh = pagemtimes(U .* reshape(ss, 1, K, nb), 'none', V, 'ctranspose');   % M x N x nb
    Xh = permute(Xh, [3 2 1]);              % nb x N x M
    for m = 1:M
        im = idx(:, m);                     % distinct corners -> no collisions for a fixed offset
        acc(im, :) = acc(im, :) + Xh(:, :, m);
        cnt(im) = cnt(im) + 1;
        s2a(im) = s2a(im) + sigma2(:);
        pa(im)  = pa(im)  + cast(Pk(:), cls);
    end
end

skipped = cnt == 0;
acc(skipped, :) = D(skipped, :);
s2a(skipped) = NaN; pa(skipped) = NaN; cnt(skipped) = 1;
denoised = reshape(double(acc ./ cnt), dims);
Sigma2   = reshape(double(s2a ./ cnt), dvox);
P        = reshape(double(pa ./ cnt), dvox);
SNR_gain = sqrt(M*N ./ (P.^2 + (M - P).*P + (N - P).*P));
info = struct('time_s', toc(t0), 'npatch', np, 'precision', cls, 'batch', opt.batch);
end
