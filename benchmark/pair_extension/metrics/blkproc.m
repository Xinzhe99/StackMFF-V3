function B = blkproc(A, blockSize, border, fun)
%BLKPROC Apply a function to distinct blocks of an image (legacy interface).
%
% The evaluation server runs base MATLAB without the Image Processing Toolbox,
% where blkproc normally lives. The vendored Chen-Varshney metric calls it in
% exactly one form:
%
%   ramda = blkproc(image, [w w], [0 0], @(x) sum(sum(x.^alpha)));
%
% Documented blkproc semantics, reproduced here:
%   * the image is divided into distinct M-by-N blocks;
%   * if the image is not an exact multiple of the block size, the trailing
%     partial block is PADDED WITH ZEROS to a full block (it is not truncated);
%   * a zero border adds no context around a block;
%   * FUN receives each full block and its return value is placed in the output,
%     so the output size is ceil(size(A)./blockSize) for scalar-returning FUN.
%
% Non-zero borders and non-scalar block outputs are rejected rather than being
% approximated, because the pipeline never requests them.

    if nargin ~= 4
        error('blkproc:args', 'blkproc(A, blockSize, [0 0], fun) requires four arguments');
    end
    if any(border(:) ~= 0)
        error('blkproc:border', 'Only zero border is supported, got %s', mat2str(border));
    end
    if ~isa(fun, 'function_handle')
        error('blkproc:fun', 'fun must be a function handle');
    end

    bs = double(blockSize(:).');
    if numel(bs) == 1
        bs = [bs bs];
    end
    if numel(bs) ~= 2 || any(bs < 1) || any(fix(bs) ~= bs)
        error('blkproc:block', 'Invalid block size');
    end

    [nrows, ncols] = size(A);
    outRows = ceil(nrows / bs(1));
    outCols = ceil(ncols / bs(2));
    out = cell(outRows, outCols);
    for r = 1:outRows
        r0 = (r-1) * bs(1) + 1;
        r1 = r0 + bs(1) - 1;
        for c = 1:outCols
            c0 = (c-1) * bs(2) + 1;
            c1 = c0 + bs(2) - 1;
            block = zeros(bs(1), bs(2), 'like', A);
            rr1 = min(r1, nrows);
            cc1 = min(c1, ncols);
            block(1:(rr1-r0+1), 1:(cc1-c0+1)) = A(r0:rr1, c0:cc1);
            out{r, c} = fun(block);
        end
    end

    first = out{1, 1};
    if isscalar(first)
        B = zeros(outRows, outCols);
        for r = 1:outRows
            for c = 1:outCols
                B(r, c) = out{r, c};
            end
        end
    else
        error('blkproc:output', 'Only scalar block outputs are supported');
    end
end
