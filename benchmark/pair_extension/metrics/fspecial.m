function h = fspecial(varargin)
%FSPECIAL Minimal drop-in replacement for the Image Processing Toolbox function.
%
% The evaluation server has base MATLAB only. The production metrics call this
% function in exactly two forms:
%   fspecial('gaussian', N, sigma)          -> QP (11, 1.5), QY (7, 1.5), VIF (N, N/5)
% No other filter type is used by the pair-metric pipeline, so any other request
% raises an error instead of returning a silently different kernel.
%
% Semantics reproduced from MATLAB's documented definition:
%   two-dimensional Gaussian with standard deviation SIGMA, centred, of size
%   HSIZE (a scalar gives an HSIZE-by-HSIZE matrix), normalised to unit sum.
% MATLAB builds it as a separable outer product of a 1-D Gaussian and enforces
% exactly unit sum; the same construction is used here so downstream
% normalisation steps (window./sum(window(:))) are unaffected.

    if nargin < 1
        error('fspecial:type', 'fspecial requires a filter type');
    end
    type = lower(varargin{1});
    if ~strcmp(type, 'gaussian')
        error('fspecial:type', 'Only the gaussian type is implemented, got %s', type);
    end
    if nargin < 3
        error('fspecial:args', 'fspecial(''gaussian'', hsize, sigma) requires sigma');
    end
    hsize = varargin{2};
    sigma = varargin{3};

    if numel(hsize) == 1
        hsize = [hsize hsize];
    end
    if numel(hsize) ~= 2 || any(hsize < 1)
        error('fspecial:hsize', 'Invalid hsize');
    end
    if ~isscalar(sigma) || ~(sigma > 0)
        error('fspecial:sigma', 'Invalid sigma');
    end

    % Separable construction: s = [-(n-1)/2 : (n-1)/2] / sigma, then exp(-s.^2/2).
    % Even kernel sizes are not used by the pipeline; require odd for a centred kernel.
    if any(mod(hsize, 2) ~= 1)
        error('fspecial:hsize', 'This replacement requires odd hsize values, got %s', mat2str(hsize));
    end
    rows = (-(hsize(1)-1)/2 : (hsize(1)-1)/2)';
    cols = (-(hsize(2)-1)/2 : (hsize(2)-1)/2);
    h = exp(-(rows.^2) / (2*sigma^2)) * exp(-(cols.^2) / (2*sigma^2));
    h = h / sum(h(:));
end
