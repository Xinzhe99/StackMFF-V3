function fflush(varargin)
%FFLUSH No-op stand-in used only by vendored code that supports GNU Octave.
%
% metricsQp_cached.m / myphasecong3.m contain the vendor's Octave compatibility
% test `v = version; Octave = v(1)<'5';`. That character comparison was written
% when MATLAB versions began with 7, 8 or 9; on R2025a the version string starts
% with '2', so the test evaluates true on genuine MATLAB and the loop calls
% fflush(1), which base MATLAB does not provide.
%
% This file restores the intended MATLAB behaviour: under MATLAB the vendor code
% never flushes explicitly, and stream flushing is automatic. It performs no
% work and cannot change any computed value. It is not a metric implementation
% and is not part of the provenance guard list.

end
