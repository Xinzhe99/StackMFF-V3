function out = evaluate_pair_metrics_linux(A, B, F)
% evaluate_pair_metrics  Linux variant for the pair-extension rescoring.
%
% Identical to evaluate_pair_metrics.m except for FMI_w: the Windows build uses
% the compiled window calculator fmi_windows_mex.mexw64, which cannot run on the
% Linux servers. Here the same dmey features and Nelsen formulas are evaluated by
% metricsFMI_wavelet_vectorized, whose output was validated against the compiled
% path and against the scalar reference to within 1e-8 on real images.
%
% Every other metric, the grayscale convention, the dependency list and the
% non-finite guard are unchanged. Do not edit metric sources while workers run.

    assert(isequal(size(A),size(B),size(F)), 'Source and output shapes must match');
    Ag = to_gray_uint8(A);
    Bg = to_gray_uint8(B);
    Fg = to_gray_uint8(F);
    Ag = double(Ag); Bg = double(Bg); Fg = double(Fg);

    v = fullfile(fileparts(mfilename('fullpath')), 'vendor', 'MEFB_metrics');
    if exist(v, 'dir'); addpath(v); end
    addpath(fullfile(fileparts(mfilename('fullpath'))));

    out = struct();
    out.NMI  = metricsNMI(Ag, Bg, Fg);
    out.FMI_w = metricsFMI_wavelet_vectorized(Ag, Bg, Fg);
    out.Qabf = metricsQabf_corrected(Ag, Bg, Fg);
    out.QP   = metricsQp_cached(Ag, Bg, Fg);
    out.QY   = metricsQy_stable(Ag, Bg, Fg);
    out.QCB  = metricsQcb(Ag, Bg, Fg);
    out.QCV  = metricsQcv(Ag, Bg, Fg);
    out.VIFF = metricsVIF(Ag, Bg, Fg);

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
        g = rgb2gray(im);
    else
        g = im;
    end
end
