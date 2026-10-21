function out = evaluate_pair_metrics(A, B, F)
% evaluate_pair_metrics  Reference fusion metrics for ONE image pair.
%   out = evaluate_pair_metrics(A, B, F)
%   A, B : source images, F : fused image.  uint8, RGB or grayscale,
%   all same size.  Returns struct with fields:
%     NMI, FMI_w, Qabf, QP, QY, QCB, QCV, VIFF
%   All higher-is-better EXCEPT QCV (lower-is-better).
%
% Conventions (fixed by lead):
%   - ONE common grayscale conversion: MATLAB rgb2gray (ITU-R BT.601)
%     applied to uint8, then double() -> intensity range 0..255.
%   - Implementations are the MEFB vendor files and reviewed derivatives
%     (work/pair_eval_2026-10-19/metrics/vendor/MEFB_metrics),
%     repo xingchenzhang/MEFB @ 9e185bf88de280b12a2d9a2ff12d66e40bc1ff91,
%     which themselves vendor zhengliu6699/imageFusionMetrics
%     @ d6f7f1a0c0a13e7d19bc1e8e4d4f3ad51f439273.
%   - FMI uses equivalence-tested compiled windows; QP caches unchanged source
%     features; Qabf fixes its relative-gradient equality branch; QY removes
%     roundoff in local variances before applying the author's formula.
%     See provenance.json and the recorded validation cases.
%
% Dependencies: Wavelet Toolbox (dwt2 'dmey' for FMI_w).
% Image Processing Toolbox is used for rgb2gray and image operations.

    assert(isequal(size(A),size(B),size(F)), 'Source and output shapes must match');
    % ---- common grayscale conversion, uint8 -> gray -> double 0..255
    Ag = to_gray_uint8(A);
    Bg = to_gray_uint8(B);
    Fg = to_gray_uint8(F);
    Ag = double(Ag); Bg = double(Bg); Fg = double(Fg);

    % ---- vendor implementations (files added on path by caller or below)
    v = fullfile(fileparts(mfilename('fullpath')), 'vendor', 'MEFB_metrics');
    if exist(v, 'dir'); addpath(v); end
    vw = fullfile(fileparts(mfilename('fullpath')));  % FMI_w wrapper here
    addpath(vw);

    out = struct();
    out.NMI  = metricsNMI(Ag, Bg, Fg);           % Hossny 2008
    out.FMI_w = metricsFMI_wavelet_mex(Ag, Bg, Fg); % validated compiled dmey variant
    out.Qabf = metricsQabf_corrected(Ag, Bg, Fg); % documented equality-branch correction
    out.QP   = metricsQp_cached(Ag, Bg, Fg);     % identical QP with exact source-feature memoization
    out.QY   = metricsQy_stable(Ag, Bg, Fg);     % Yang, with documented variance stability repair
    out.QCB  = metricsQcb(Ag, Bg, Fg);           % Chen-Blum
    out.QCV  = metricsQcv(Ag, Bg, Fg);           % Chen-Varshney (LOWER is better)
    out.VIFF = metricsVIF(Ag, Bg, Fg);           % fusion-specific VIFF (Han 2013)

    % nonfinite guard: propagate NaN with a flag rather than crash silently
    f = fieldnames(out);
    out.nonfinite = {};
    for i = 1:numel(f)
        if ~isfinite(out.(f{i}))
            out.nonfinite{end+1} = f{i}; %#ok<AGROW>
        end
    end
end

function g = to_gray_uint8(im)
    assert(isa(im, 'uint8'), 'evaluate_pair_metrics: inputs must be uint8');
    if size(im, 3) == 3
        g = rgb2gray(im);   % ITU-R BT.601, uint8
    else
        g = im;
    end
end
