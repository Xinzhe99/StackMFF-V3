function score=metricsFMI_wavelet_mex(A,B,F)
% Same MATLAB wavelet features; validated compiled scalar window calculation.
score=fmi_windows_mex(feature(A),feature(B),feature(F));
end
function X=feature(X)
[a,h,v,d]=dwt2(double(X),'dmey');X=[a,h;v,d];
lo=min(X(:));hi=max(X(:));
if hi==lo,X=ones(size(X));else,X=(X-lo)/(hi-lo);end
end
