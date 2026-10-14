function h = fspecial(type, n, sigma)
% Minimal stand-in for MATLAB fspecial (Image Processing Toolbox), 'gaussian' only,
% following MATLAB's documented algorithm (Octave image pkg not installed).
if ~strcmp(type, 'gaussian'), error('only gaussian'); end
if numel(n) == 1, n = [n n]; end
siz = (n - 1) / 2;
[x, y] = meshgrid(-siz(2):siz(2), -siz(1):siz(1));
h = exp(-(x.*x + y.*y) / (2 * sigma * sigma));
h(h < eps * max(h(:))) = 0;
s = sum(h(:));
if s ~= 0, h = h / s; end
end
