function m = mean2(A)
%MEAN2 Mean of all elements in a matrix.
%
% The evaluation server runs base MATLAB without the Image Processing
% Toolbox, where mean2 is normally provided. Vendored metric code calls it on
% double matrices. MATLAB's documented definition is the arithmetic mean of all
% elements, which is exactly mean(A(:)). The result is computed in double
% precision, matching the toolbox behaviour for the numeric classes used here.

    if ~isnumeric(A) && ~islogical(A)
        error('mean2:type', 'mean2 expects a numeric or logical array');
    end
    m = mean(A(:));
end
