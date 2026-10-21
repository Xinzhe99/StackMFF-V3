function run_metric_worker_linux(dataRoot, resultRoots, scoreRoot, lane, nlanes, manifestPath)
% Score every complete, readable output exactly once on this host.
% Linux counterpart of run_metric_worker.m: identical scan contract, identical
% output record schema, but it evaluates through evaluate_pair_metrics_linux so
% FMI_w does not require the compiled Windows helper.
%
% One lane owns each method/scene. A missing output stays missing; it never
% becomes a zero or a ranked subset. Existing records are skipped, and an
% existing record is never overwritten (the aggregator rejects disagreements).

maxNumCompThreads(2);
here = fileparts(mfilename('fullpath'));
addpath(here);

manifest = jsondecode(fileread(manifestPath));
methods = unique({manifest.methods.name});
datasets = {'Lytro', 'MFFW', 'MFI-WHU', 'RealMFF'};
counts = [20, 13, 120, 710];
expected = containers.Map();
for k = 1:numel(manifest.scenes)
    s = manifest.scenes(k);
    expected([s.dataset '/' s.scene]) = [s.a.width, s.a.height];
end

if ischar(resultRoots), resultRoots = {resultRoots}; end
assert(lane >= 0 && lane < nlanes);
lastProgress = tic;
while toc(lastProgress) < 48*3600
    task = 0; progress = 0; remaining = 0;
    for d = 1:numel(datasets)
        ds = datasets{d};
        for id = 1:counts(d)
            task = task + 1;
            if mod(task-1, nlanes) ~= lane, continue; end
            for m = 1:numel(methods)
                method = methods{m};
                sid = num2str(id);
                outPath = char(fullfile(scoreRoot, method, ds, [sid '.json']));
                if exist(outPath, 'file'), continue; end
                remaining = remaining + 1;
                filename = [sid '.png'];
                if strcmp(method, 'SwinFusion'), filename = [sid '_rgb.png']; end
                candidates = cellfun(@(root) fullfile(root, method, ds, filename), resultRoots, 'UniformOutput', false);
                candidates = candidates(cellfun(@(p) exist(p, 'file') == 2, candidates));
                if isempty(candidates), continue; end
                fp = candidates{1};
                for q = 2:numel(candidates)
                    assert(strcmp(sha256_file(fp), sha256_file(candidates{q})), 'Conflicting duplicate output');
                end
                info = dir(fp);
                if isempty(info), continue; end
                if (now - info.datenum) * 86400 < 5, continue; end
                ap = fullfile(dataRoot, ds, 'image stack', sid, '1.jpg');
                bp = fullfile(dataRoot, ds, 'image stack', sid, '2.jpg');
                try
                    A = imread(ap); B = imread(bp); F = imread(fp);
                    assert(isa(A,'uint8') && isa(B,'uint8') && isa(F,'uint8') && size(F,3) == 3);
                    assert(isequal(size(A), size(B), size(F)));
                    key = [ds '/' sid];
                    want = expected(key);
                    assert(size(F,2) == want(1) && size(F,1) == want(2), 'Output dimensions differ from the manifest');
                    t = tic;
                    scores = evaluate_pair_metrics_linux(A, B, F);
                    assert(isempty(scores.nonfinite), 'Nonfinite metric');
                    names = fieldnames(scores);
                    for k = 1:numel(names)
                        if strcmp(names{k}, 'nonfinite'), continue; end
                        v = scores.(names{k});
                        assert(isscalar(v));
                        assert(abs(imag(v)) < 1e-10*max(1, abs(real(v))), 'Unexpected complex metric');
                        scores.(names{k}) = real(v);
                    end
                    record = struct('method', method, 'dataset', ds, 'scene', sid, 'metrics', scores, ...
                        'seconds', toc(t), 'width', size(F,2), 'height', size(F,1), 'output_sha256', sha256_file(fp), ...
                        'input_a_sha256', sha256_file(ap), 'input_b_sha256', sha256_file(bp), ...
                        'matlab_version', version, 'implementation_manifest_sha256', sha256_file(fullfile(here, 'provenance.json')));
                    outPath = char(outPath);
                    outDir = char(fileparts(outPath));
                    if ~exist(outDir, 'dir'), mkdir(outDir); end
                    partPath = char(strcat(outPath, '.part'));
                    fid = fopen(partPath, 'w');
                    assert(fid >= 0);
                    fprintf(fid, '%s', jsonencode(record));
                    fclose(fid);
                    movefile(partPath, outPath);
                    progress = progress + 1;
                    fprintf('SCORED %s %s/%s %.3fs\n', method, ds, sid, record.seconds);
                catch err
                    errDir = char(fullfile(scoreRoot, 'errors'));
                    if ~exist(errDir, 'dir'), mkdir(errDir); end
                    er = char(fullfile(errDir, sprintf('%s_%s_%s.txt', char(method), char(ds), char(sid))));
                    fid = fopen(er, 'w');
                    if fid >= 0
                        fprintf(fid, '%s', getReport(err, 'extended', 'hyperlinks', 'off'));
                        fclose(fid);
                    end
                    fprintf('METRIC_ERROR %s %s/%s %s\n', method, ds, sid, err.message);
                end
            end
        end
    end
    if progress > 0, lastProgress = tic; end
    if remaining == 0, fprintf('METRIC_LANE_COMPLETE %d\n', lane); return; end
    if progress == 0, pause(60); end
end
fprintf('METRIC_LANE_IDLE_TIMEOUT %d; incomplete inputs remain\n', lane);
end

function h = sha256_file(p)
fid = fopen(p, 'rb');
assert(fid >= 0);
bytes = fread(fid, Inf, '*uint8');
fclose(fid);
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(bytes);
h = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end
