function validate_linux_metrics(cfgPath, outPath)
% Score a fixed validation cohort with the Linux metric path and write one
% JSON record per row. Used only to prove equivalence with the archived
% Windows scores; it does not modify any production record.

addpath(fileparts(mfilename('fullpath')));
addpath(fullfile(fileparts(mfilename('fullpath')), 'vendor', 'MEFB_metrics'));
cfg = jsondecode(fileread(cfgPath));
W = cfg.root;
out = {};
for k = 1:numel(cfg.rows)
    r = cfg.rows(k);
    fp = fullfile(W, 'results', r.method, r.dataset, [r.scene '.png']);
    if ~exist(fp, 'file')
        fp = fullfile(W, 'results', r.method, r.dataset, [r.scene '_rgb.png']);
    end
    ap = fullfile(W, 'inputs', r.dataset, 'image stack', r.scene, '1.jpg');
    bp = fullfile(W, 'inputs', r.dataset, 'image stack', r.scene, '2.jpg');
    try
        A = imread(ap); B = imread(bp); F = imread(fp);
        s = evaluate_pair_metrics_linux(A, B, F);
        out{k} = struct('method', r.method, 'dataset', r.dataset, 'scene', r.scene, ...
            'FMI_w', s.FMI_w, 'NMI', s.NMI, 'Qabf', s.Qabf, 'QP', s.QP, 'QY', s.QY, ...
            'QCB', s.QCB, 'QCV', s.QCV, 'VIFF', s.VIFF, 'ok', true, 'error', '', ...
            'width', size(F, 2), 'height', size(F, 1)); %#ok<AGROW>
    catch e
        out{k} = struct('method', r.method, 'dataset', r.dataset, 'scene', r.scene, ...
            'FMI_w', NaN, 'NMI', NaN, 'Qabf', NaN, 'QP', NaN, 'QY', NaN, 'QCB', NaN, ...
            'QCV', NaN, 'VIFF', NaN, 'ok', false, 'error', e.message, ...
            'width', 0, 'height', 0); %#ok<AGROW>
    end
end
fid = fopen(outPath, 'w');
fprintf(fid, '%s', jsonencode(out));
fclose(fid);
fprintf('VALIDATION_DONE %d\n', numel(out));
end
