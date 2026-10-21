function [ca, ch, cv, cd] = dwt2(x, varargin)
%DWT2 Single-level 2-D discrete wavelet transform for the 'dmey' wavelet.
%
% The evaluation server has base MATLAB only: the Wavelet Toolbox is absent, so
% dwt2 is not available there. The pair-metric pipeline uses exactly one call
% form, [a,h,v,d] = dwt2(double(X), 'dmey'), with MATLAB's default extension
% mode. This file reproduces that call from the version-controlled coefficient
% table in dmey.mat, following MathWorks dwt2/dwt/wfilters/orthfilt line by line:
%
%   1. filters: W = dmey(:); W = W/sum(W); Lo_R = sqrt(2)*W;
%      Hi_R = qmf(Lo_R,0); Lo_D = flip(Lo_R); Hi_D = flip(Hi_R);
%   2. default dwtmode is 'sym' (half-point symmetric) with shift 0, so
%      first = 2, last = size + lf - 1 and the extension length is lf - 1;
%   3. dwt2 first extends COLUMNS along the row index ('addcol'), convolves
%      rows with 'valid' and then calls convdown, which decimates columns
%      starting at `first`, extends ROWS along the column index ('addrow'),
%      convolves columns with 'valid' and decimates rows starting at `first`.
%
% Any other wavelet name, extension mode, shift or explicit filter pair raises
% an error instead of silently returning a different transform.

    if nargin < 2
        error('dwt2:args', 'dwt2(x, wname) requires a wavelet name');
    end
    if ~(ischar(varargin{1}) || isstring(varargin{1}))
        error('dwt2:args', 'Only the dwt2(x, ''dmey'') form is supported on this host');
    end
    if ~strcmpi(char(varargin{1}), 'dmey')
        error('dwt2:wavelet', 'Only the dmey wavelet is implemented, got %s', char(varargin{1}));
    end
    if numel(varargin) > 1
        error('dwt2:args', 'Extension mode, shift and explicit filters are not supported');
    end

    [Lo_D, Hi_D] = dmey_filters();
    first = 2;
    x = double(x);
    if ndims(x) > 2 && size(x, 3) ~= 3
        error('dwt2:shape', 'Expected a matrix or a three-channel image');
    end

    if size(x, 3) == 3
        ca = []; ch = []; cv = []; cd = [];
        for k = 1:3
            [a, h, v, d] = one_level(x(:, :, k), Lo_D, Hi_D, first);
            ca = cat(3, ca, a); ch = cat(3, ch, h);
            cv = cat(3, cv, v); cd = cat(3, cd, d);
        end
    else
        [ca, ch, cv, cd] = one_level(x, Lo_D, Hi_D, first);
    end
end

function [a, h, v, d] = one_level(x, Lo_D, Hi_D, first)
    lf = numel(Lo_D);
    ext = lf - 1;
    % dwt2 body for a single-channel image.
    y = sym_extend(x, 2, ext);                 % addcol: extend the column index
    z = conv2(y, Lo_D(:)', 'valid');
    a = convdown(z, Lo_D, ext, first);
    h = convdown(z, Hi_D, ext, first);
    z = conv2(y, Hi_D(:)', 'valid');
    v = convdown(z, Lo_D, ext, first);
    d = convdown(z, Hi_D, ext, first);
end

function y = convdown(x, f, len_ext, first)
    % MathWorks convdown: decimate columns, extend rows, convolve, decimate rows.
    y = x(:, first:2:end);
    y = sym_extend(y, 1, len_ext);             % addrow: extend the row index
    y = conv2(y', f(:)', 'valid');
    y = y';
    y = y(first:2:end, :);
end

function y = sym_extend(x, dim, len)
    % Half-point symmetric extension of x along dimension dim by len samples,
    % matching MathWorks wextend('addcol'/'addrow','sym',...). In wextend,
    % 'addcol' extends the second dimension and 'addrow' the first.
    n = size(x, dim);
    idx = sym_indices(n, len);
    if dim == 1
        y = x(idx, :, :);
    else
        y = x(:, idx, :);
    end
end

function idx = sym_indices(n, len)
    % Half-point symmetric index sequence: ...,2,1 | 1..n | n,n-1,...
    base = [len:-1:1, 1:n, n:-1:n-len+1];
    period = 2 * n;
    base = mod(base - 1, period) + 1;
    flip = base > n;
    base(flip) = 2 * n + 1 - base(flip);
    idx = base;
end

function [Lo_D, Hi_D] = dmey_filters()
    persistent LO HI
    if isempty(LO)
        here = fileparts(mfilename('fullpath'));
        S = load(fullfile(here, 'dmey.mat'), 'dmey');
        W = S.dmey(:);
        W = W / sum(W);
        Lo_R = sqrt(2) * W;
        Hi_R = qmf_local(Lo_R, 0);
        LO = Lo_R(end:-1:1);
        HI = Hi_R(end:-1:1);
    end
    Lo_D = LO; Hi_D = HI;
end

function y = qmf_local(x, p)
    y = x(end:-1:1);
    firstIdx = 2 - rem(p, 2);
    y(firstIdx:2:end) = -y(firstIdx:2:end);
end
