function [denoised,Sigma2,P,SNR_gain,info] = denoise_recursive_tensor_batched(data,window,varargin)
% Batched (vectorised) version of denoise_recursive_tensor: same inputs, options and outputs.
%
%   [denoised,Sigma2,P,SNR_gain,info] = denoise_recursive_tensor_batched(data,window,varargin)
%
% Same algorithm and options as denoise_recursive_tensor / denoise_array_recursive_tensor
% (indices, mask, stride, center_assign, opt_shrink, sigma, full_sigma2_pass, num_inds),
% but all sliding-window patches of a batch are processed together: every SVD is a pagesvd
% over the patches (multithreaded across pages) instead of one svd per patch in a loop.
% In the tensor case the number of retained components P differs between patches; patches
% are zero-padded to full rank and each patch's true matrix sizes are tracked, so the
% Marchenko-Pastur cutoffs and optimal shrinkage are exactly those of the original code.
%
% Extra options:  'batch'     patches per batch (default 20000)
%                 'precision' 'double' (default, matches the original to rounding error) or 'single'
% Not batched:    'subtract_mean' = true -> falls back to denoise_recursive_tensor (with a warning).
% Extra output:   info.time_s, info.npatch, info.fallback
%
% Please cite: Olesen JL, Ianus A, Østergaard L, Shemesh N, Jespersen SN. Tensor denoising of
% multidimensional MRI data. Magn Reson Med. 2022. doi:10.1002/mrm.29478
%
% Derived from denoise_recursive_tensor.m / denoise_array_recursive_tensor.m of Tensor-MP-PCA
% (https://github.com/Neurophysics-CFIN/Tensor-MP-PCA). Modified by Kwok-Shing Chan (MGH),
% November 2026: batched pagesvd implementation of the full recursive-tensor algorithm.
%
% This program is free software: you can redistribute it and/or modify it under the terms of
% the GNU General Public License as published by the Free Software Foundation, either version 3
% of the License, or (at your option) any later version. This program is distributed in the hope
% that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
% (LICENSE) for more details.

t0 = tic;
% ---- options (same defaults as the original) ----
dims = size(data);
dims_vox = dims(1:length(window));
if numel(dims_vox)==1, dims_vox(2) = 1; end
num_vox = prod(dims_vox);

vox_indices = 1:length(window);
mod_indices = length(window)+1:length(dims);
options.indices = cat(2,{vox_indices},num2cell(mod_indices));
options.mask = true(dims_vox);
options.center_assign = false;
options.stride = ones(1,length(window));
options.test = false;
options.batch = 20000;
options.precision = 'double';
% denoise_array_recursive_tensor options (passed through by the original via varargin)
aopt.opt_shrink = true; aopt.subtract_mean = false; aopt.full_sigma2_pass = true;
aopt.num_inds = []; aopt.sigma2 = [];
array_opts = {'opt_shrink','subtract_mean','full_sigma2_pass','num_inds','sigma2'};
for n = 1:2:length(varargin)
    if ismember(varargin{n}, array_opts), aopt.(varargin{n}) = varargin{n+1}; end
    options.(varargin{n}) = varargin{n+1};
end

if aopt.subtract_mean
    warning('denoise_recursive_tensor_batched:fallback', 'subtract_mean is not batched; using denoise_recursive_tensor.');
    keep = ~ismember(varargin(1:2:end), {'batch','precision'});
    va = reshape(varargin, 2, []); va = va(:, keep); va = va(:)';
    [denoised,Sigma2,P,SNR_gain] = denoise_recursive_tensor(data,window,va{:});
    info = struct('time_s', toc(t0), 'npatch', NaN, 'fallback', true);
    return
end

data = reshape(data,num_vox,[]);
indices = options.indices;
stride = reshape(options.stride,1,[]);
assert(all(size(options.mask)==dims_vox),'mask dimensions do not match data dimensions');

window = reshape(window,1,[]);
array_size = [window dims(length(window)+1:end)];
array_size = cat(2,array_size,ones(1,max(cell2mat(indices))-length(dims)));

window_subs = cell(1,length(window));
[window_subs{:}] = ind2sub(window,1:prod(window));
index_increments = -1 + sub2ind(dims,window_subs{:});         % 1 x W

permute_order = cell2mat(indices);
permute_order = cat(2,permute_order,setdiff(1:length(array_size),permute_order));

new_size = zeros(1,numel(indices)+1);
for n = 1:numel(indices), new_size(n) = prod(array_size(indices{n})); end
new_size(end) = prod(array_size)/prod(new_size(1:end-1));
new_size(new_size==1) = [];
new_size(end+1) = 1;
core = new_size(1:end-1);                 % dimensions of each denoised array (size(X) without trailing 1)
if isscalar(core), core = [core 1]; end
num_inds_patch = min(numel(core), numel(indices));
if ~isempty(aopt.num_inds), num_inds_patch = aopt.num_inds; end   % explicit 'num_inds' overrides, as in the original

% ---- window positions (same rules as the original loop) ----
subs = cell(1,numel(dims_vox));
[subs{:}] = ind2sub(dims_vox,(1:num_vox)');
corner_sub = cell2mat(subs);                                  % num_vox x ndims
ok = all(corner_sub-1+window <= dims_vox, 2) & all(rem(corner_sub-1, stride)==0, 2);
corners = find(ok);
W = numel(index_increments);
vidx = corners + index_increments;                            % npos x W
inmask = options.mask(vidx);
ok = any(inmask, 2);
if options.center_assign
    center_subs = num2cell(ceil(window/2));
    center_ind = sub2ind(window,center_subs{:});
    ok = ok & inmask(:,center_ind);
end
corners = corners(ok);
np = numel(corners);

cls = options.precision;
data_c = cast(data, cls);
Nm = size(data,2);
denoised = zeros(size(data), cls);
count = zeros(num_vox,1,cls);
Sig = zeros(num_vox,1,cls);
Pacc = zeros(num_vox, 1, cls);

for b0 = 1:options.batch:np
    c = corners(b0:min(np, b0+options.batch-1)); nb = numel(c);
    idx = c + index_increments;                               % nb x W
    % patches: [nb, array_size] -> permute -> [core..., nb]
    Xb = reshape(data_c(idx(:),:), [nb array_size]);
    Xb = permute(Xb, [1 permute_order+1]);
    Xb = reshape(Xb, [nb core]);
    Xb = permute(Xb, [2:numel(core)+1 1]);                    % [core..., nb]
    [Xb, s2, p] = denoise_array_batched(Xb, core, num_inds_patch, aopt, cls);
    % back to [nb*W, Nm] in idx(:) order
    Xb = permute(Xb, [numel(core)+1 1:numel(core)]);          % [nb, core...]
    Xb = reshape(Xb, [nb array_size(permute_order)]);
    Xb = ipermute(Xb, [1 permute_order+1]);
    Xb = reshape(Xb, nb, W, Nm);
    if size(p,1) > size(Pacc,2), Pacc(:, end+1:size(p,1)) = 0; end
    if options.center_assign
        im = idx(:, center_ind);
        denoised(im,:) = denoised(im,:) + reshape(Xb(:,center_ind,:), nb, Nm);
        count(im) = count(im) + 1; Sig(im) = Sig(im) + s2(:);
        Pacc(im,1:size(p,1)) = Pacc(im,1:size(p,1)) + p';
    else
        for m = 1:W                                           % distinct corners: no collisions for a fixed offset
            im = idx(:, m);
            denoised(im,:) = denoised(im,:) + reshape(Xb(:,m,:), nb, Nm);
            count(im) = count(im) + 1; Sig(im) = Sig(im) + s2(:);
            Pacc(im,1:size(p,1)) = Pacc(im,1:size(p,1)) + p';
        end
    end
end

% ---- assign skipped voxels and average (as the original) ----
denoised = double(denoised); count = double(count); Sigma2 = double(Sig); P = double(Pacc);
skipped_vox = count==0;
denoised(skipped_vox,:) = data(skipped_vox,:);
Sigma2(skipped_vox) = nan;
P(skipped_vox,:) = nan;
count(skipped_vox) = 1;
denoised = denoised./count;
Sigma2 = Sigma2./count;
P = P./count;

% Note: when 'num_inds' < number of indices, the original's SNR_gain line errors (P has fewer
% columns than new_size); here the indices that are not denoised count as full rank (P = their size).
Pfull = P;
if size(P,2) > 1 && size(P,2) < numel(new_size)-1
    Pfull = [P, repmat(new_size(size(P,2)+1:end-1), size(P,1), 1)];
end
if size(Pfull,2)==1
    SNR_gain = sqrt(prod(new_size)./(Pfull.^2+sum((new_size(1:end-1)-Pfull).*Pfull,2)));
else
    SNR_gain = sqrt(prod(new_size)./(prod(Pfull,2)+sum((new_size(1:end-1)-Pfull).*Pfull,2)));
end

denoised = reshape(denoised,dims);
Sigma2 = reshape(Sigma2,dims_vox);
P = reshape(P,[dims_vox size(P,2)]);
SNR_gain = reshape(SNR_gain,dims_vox);
info = struct('time_s', toc(t0), 'npatch', np, 'fallback', false);
end


% =========================================================================================
function [X, sigma2, P] = denoise_array_batched(X, dims, num_inds, aopt, cls)
% Batched denoise_array_recursive_tensor. X: [dims..., nb]. Returns X (same size),
% sigma2 (1 x nb) and P (num_P x nb; 1 row in the matrix case, num_inds rows otherwise).
nb = size(X, numel(dims)+1);
use_MPPCA = isempty(aopt.sigma2);
tot = prod(dims);

%% matrix case
if numel(dims) == 2
    if min(dims) == 1
        sigma2 = zeros(1,nb,cls); P = ones(1,nb,cls); return
    end
    [U,S,V] = pagesvd(reshape(X, dims(1), dims(2), nb), 'econ');
    s2 = pagediag2(S);
    if use_MPPCA
        sigma2 = estimate_noise_b(s2, tot);
    else
        sigma2 = cast(aopt.sigma2, cls) * ones(1,nb,cls);
    end
    keepc = s2 > sigma2 .* (sqrt(dims(1))+sqrt(dims(2))).^2;
    P = sum(keepc, 1);
    ss = sqrt(s2) .* keepc;
    if aopt.opt_shrink
        ss = shrink_b(s2, keepc, P, dims(1), dims(2), sigma2);
    end
    X = reshape(pagemtimes(U .* reshape(ss,1,[],nb), 'none', V, 'ctranspose'), [dims nb]);
    P = cast(P, cls);
    return
end

%% tensor case
U = cell(1,num_inds); V = U; s2c = U; Kn = zeros(1,num_inds);
if use_MPPCA
    if aopt.full_sigma2_pass, num_SVDs = num_inds; else, num_SVDs = 1; end
    Y = reshape(X, tot, nb);
    Pn = zeros(num_SVDs, nb, cls);
    for n = 1:num_SVDs
        Yn = reshape(Y, dims(n), [], nb);
        [U{n},S,V{n}] = pagesvd(Yn, 'econ');
        s2c{n} = pagediag2(S); Kn(n) = size(S,1);
        [~, Pn(n,:)] = estimate_noise_b(s2c{n}, tot);            % initial individual P estimates
        Y = reshape(pagemtimes(V{n}, S), [], nb);                % X = V*S (full, no truncation in this pass)
    end
    % combined_noise_estimate
    num = zeros(1,nb,cls); den = zeros(1,nb,cls);
    for n = 1:num_SVDs
        M = Kn(n); N = tot/M;
        tail = flipud(cumsum(flipud(s2c{n}),1));                 % tail(k) = sum(s2(k:end))
        tail = [tail; zeros(1,nb,cls)];
        num = num + tail(sub2ind(size(tail), Pn(n,:)+1, 1:nb));
        den = den + (M - Pn(n,:)) .* (N - Pn(n,:));
    end
    sigma2 = num ./ den;
else
    sigma2 = cast(aopt.sigma2, cls) * ones(1,nb,cls);
    [U{1},S,V{1}] = pagesvd(reshape(X, dims(1), [], nb), 'econ');
    s2c{1} = pagediag2(S); Kn(1) = size(S,1);
end

% recursive truncation (zero-padded to full rank; true sizes tracked per patch)
P = zeros(num_inds, nb, cls);
Nact = tot/dims(1) * ones(1,nb,cls);                             % true number of columns of the n-th unfolding
% n = 1: reuse the first SVD; never shrunk here (the original shrinks only inside its n>=2 loop)
[~, P(1,:), ss] = discard_b(s2c{1}, dims(1), Nact, sigma2, false);
X = reshape(V{1} .* reshape(ss,1,[],nb), [], nb);                % V*S_P (columns > P are zero)
for n = 2:num_inds
    Nact = Nact .* P(n-1,:) / dims(n);                            % true columns of the n-th unfolding
    Xn = reshape(X, dims(n), [], nb);
    [U{n},S,V{n}] = pagesvd(Xn, 'econ');
    s2 = pagediag2(S); Kn(n) = size(S,1);
    [~, P(n,:), ss] = discard_b(s2, dims(n), Nact, sigma2, aopt.opt_shrink && n==num_inds);
    X = reshape(V{n} .* reshape(ss,1,[],nb), [], nb);
end
% patches where some P(n-1)==0 stopped there in the original (rest of P = 0, output 0):
% here their X is identically zero from that step on, so the result is the same.

% reconstruct (backward pass): X = U{n} * reshape(X,[],K_n)'
for n = num_inds:-1:1
    Xr = reshape(X, [], Kn(n), nb);
    X = reshape(pagemtimes(U{n}, 'none', Xr, 'transpose'), [], nb);
end
X = reshape(X, [dims nb]);
end

% -----------------------------------------------------------------------------------------
function s2 = pagediag2(S)
% squared diagonal of each page of S (K x K x nb) -> K x nb
K = size(S,1); nb = size(S,3);
lin = (0:K-1)*(K+1) + 1;
S2 = reshape(S, K*K, nb);
s2 = S2(lin, :).^2;
end

function [sigma2, P] = estimate_noise_b(s2, tot)
% estimate_noise of the original, per page; M = number of singular values, N = prod(dims)/M
[K, nb] = size(s2); M = K; N = tot/M;
kk = cast((0:K-1)', 'like', s2);
sig_est = flipud(cumsum(flipud(s2),1)) ./ ((M - kk) .* (N - kk));
below = s2 < sig_est .* (sqrt(M)+sqrt(N)).^2;
[hasP, first] = max(below, [], 1);
P = first - 1;
sigma2 = sig_est(sub2ind([K nb], first, 1:nb));
sigma2(~hasP) = 0; P(~hasP) = K;
if min(M,N) == 1, P(P==0) = 1; end
P = cast(P, 'like', s2);
end

function [keepc, P, ss] = discard_b(s2, M, N, sigma2, do_shrink)
% discard_noise_components (+ apply_optimal_shrinkage when do_shrink) per page.
% M = rows of U (dims(n)), N = TRUE rows of V for each page (1 x nb).
keepc = s2 > sigma2 .* (sqrt(M) + sqrt(N)).^2;
P = sum(keepc, 1);
ss = sqrt(s2) .* keepc;
if do_shrink
    ss = shrink_b(s2, keepc, P, M, N, sigma2);
end
P = cast(P, 'like', s2);
end

function ss = shrink_b(s2, keepc, P, M, N, sigma2)
% apply_optimal_shrinkage / opt_shrink_frob on the kept components
Mp = max(M - P, 1); Np = max(N - P, 1);
v = s2 - 2*(Np + Mp).*sigma2 + (Np - Mp).^2 .* sigma2.^2 ./ s2;
ss = real(sqrt(complex(v))) .* keepc;
ss(~keepc) = 0;
end
