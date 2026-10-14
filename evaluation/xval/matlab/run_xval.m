% Cross-validation driver: reference .m implementations on the exported gray cases.
addpath(pwd);
fid0 = fopen('../cases/cases.txt'); C = textscan(fid0, '%s %s %d'); fclose(fid0);
out = fopen('matlab_results.csv', 'w');
fprintf(out, 'dir,method,SSIM,PSNR,VIF,MI,NMI,Qabf2_crop,Xydeas2_crop,MEF_crop\n');
for c = 1:numel(C{1})
    d = ['../cases/' C{1}{c} '/']; m = C{2}{c}; n = C{3}(c);
    gt = imread([d 'gt.png']); f = imread([d 'fused_' m '.png']);
    s = ssim_index(double(gt), double(f));
    mse = mean((double(gt(:)) - double(f(:))).^2); p = 10*log10(255^2/mse);
    v = vifp_mscale(gt, f);
    [mi, hxy, hx, hy] = mutual_info(gt, f); nmi = 2*mi/(hx+hy);
    fc = double(imread([d 'fused_crop_' m '.png']));
    L1 = double(imread(sprintf('%slayer_crop_%02d.png', d, 1)));
    L2 = double(imread(sprintf('%slayer_crop_%02d.png', d, 2)));
    q = Qabf(L1, L2, fc);
    qx = metricXydeas(L1, L2, fc);
    seq = zeros(size(fc,1), size(fc,2), n);
    for k = 1:n, seq(:,:,k) = double(imread(sprintf('%slayer_crop_%02d.png', d, k))); end
    me = mef_ssim(seq, fc);
    fprintf(out, '%s,%s,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g,%.15g\n', C{1}{c}, m, s, p, v, mi, nmi, q, qx, me);
    fprintf('%d/%d %s %s done\n', c, numel(C{1}), C{1}{c}, m); 
end
fclose(out);
