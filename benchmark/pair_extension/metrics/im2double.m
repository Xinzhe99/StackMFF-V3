function y = im2double(x)
%IM2DOUBLE Convert an image to double precision.
%
% Image Processing Toolbox function, absent from base MATLAB on the evaluation
% server. MATLAB's documented mapping, which the vendored metric code depends on:
%   uint8   -> double(x)/255
%   uint16  -> double(x)/65535
%   double  -> unchanged
%   single  -> double(x)
% The metrics only ever pass uint8 or double arrays, which are handled exactly;
% other integer classes raise an error rather than being silently rescaled.

    if isa(x, 'double')
        y = x;
    elseif isa(x, 'single')
        y = double(x);
    elseif isa(x, 'uint8')
        y = double(x) / 255;
    elseif isa(x, 'uint16')
        y = double(x) / 65535;
    elseif islogical(x)
        y = double(x);
    else
        error('im2double:class', 'Unsupported input class %s', class(x));
    end
end
