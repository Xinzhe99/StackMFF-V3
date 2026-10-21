function run_metric_worker(dataRoot,resultRoot,scoreRoot,lane,nlanes)
% Score only complete, readable outputs. One lane owns each method/scene.
% A missing output stays missing; it never becomes a zero or a ranked subset.
maxNumCompThreads(2);
here=fileparts(mfilename('fullpath'));addpath(here);
if ischar(resultRoot),resultRoot={resultRoot};end
provenancePath=fullfile(here,'provenance.json');implementationHash=sha256_file(provenancePath);
provenance=jsondecode(fileread(provenancePath));
for g=1:numel(provenance.code_files)
 guard=provenance.code_files(g);
 assert(strcmp(sha256_file(fullfile(here,guard.path)),guard.sha256),'Metric source changed after validation');
end
methods={'GFF','IFM','QB','BFMF','GFDF','MGFF','CVT','DWT','DCT','DTCWT','NSCT', ...
 'IFCNN','U2Fusion','SDNet','MFF-GAN','SwinFusion','MUFusion','SwinMFF','DDBFusion','CCSR-Net','MCCSR-Net', ...
 'DeFusion','MFIF-GAN','GIFNet','CSNet','Helicon_A','Helicon_B','Helicon_C','Zerene_DMap','Zerene_PMax', ...
 'StackMFF','StackMFF_V2','VAEEDOF','Ours_hard','Ours_soft','ZMFF','ReDiffuse'};
datasets={'Lytro','MFFW','MFI-WHU','RealMFF'};counts=[20,13,120,710];
assert(lane>=0 && lane<nlanes);
lastProgress=tic;
while toc(lastProgress)<48*3600
    task=0;progress=0;remaining=0;
    for d=1:numel(datasets)
      ds=datasets{d};
      for id=1:counts(d)
        task=task+1;if mod(task-1,nlanes)~=lane, continue; end
       for m=1:numel(methods)
        method=methods{m};
        sid=num2str(id);result=fullfile(scoreRoot,method,ds,[sid,'.json']);
        if exist(result,'file'),continue;end
        remaining=remaining+1;
        filename=[sid,'.png'];if strcmp(method,'SwinFusion'),filename=[sid,'_rgb.png'];end
        candidates=cellfun(@(root)fullfile(root,method,ds,filename),resultRoot,'UniformOutput',false);
        candidates=candidates(cellfun(@(p)exist(p,'file')==2,candidates));
        if isempty(candidates),continue;end
        fp=candidates{1};
        for q=2:numel(candidates),assert(strcmp(sha256_file(fp),sha256_file(candidates{q})),'Conflicting duplicate output');end
        info=dir(fp);if (now-info.datenum)*86400<5,continue;end
        ap=fullfile(dataRoot,ds,'A',[sid,'.jpg']);bp=fullfile(dataRoot,ds,'B',[sid,'.jpg']);
        try
          A=imread(ap);B=imread(bp);F=imread(fp);
          assert(isa(A,'uint8') && isa(B,'uint8') && isa(F,'uint8') && size(F,3)==3);
          assert(isequal(size(A),size(B),size(F)));
          t=tic;scores=evaluate_pair_metrics(A,B,F);assert(isempty(scores.nonfinite),'Nonfinite metric');
          names=fieldnames(scores);
          for k=1:numel(names)
            if strcmp(names{k},'nonfinite'),continue;end
            v=scores.(names{k});assert(isscalar(v));
            assert(abs(imag(v))<1e-10*max(1,abs(real(v))),'Unexpected complex metric');
            scores.(names{k})=real(v);
          end
          record=struct('method',method,'dataset',ds,'scene',sid,'metrics',scores, ...
            'seconds',toc(t),'width',size(F,2),'height',size(F,1),'output_sha256',sha256_file(fp), ...
            'input_a_sha256',sha256_file(ap),'input_b_sha256',sha256_file(bp), ...
            'matlab_version',version,'implementation_manifest_sha256',implementationHash);
          folder=fileparts(result);if ~exist(folder,'dir'),mkdir(folder);end
          tmp=[result,'.part'];fid=fopen(tmp,'w');assert(fid>=0);fprintf(fid,'%s',jsonencode(record));fclose(fid);movefile(tmp,result);
          progress=progress+1;fprintf('SCORED %s %s/%s %.3fs\n',method,ds,sid,record.seconds);
        catch err
          folder=fullfile(scoreRoot,'errors');if ~exist(folder,'dir'),mkdir(folder);end
          er=fullfile(folder,sprintf('%s_%s_%s.txt',method,ds,sid));fid=fopen(er,'w');fprintf(fid,'%s',getReport(err,'extended','hyperlinks','off'));fclose(fid);
          fprintf('METRIC_ERROR %s %s/%s %s\n',method,ds,sid,err.message);
        end
       end
      end
    end
    if progress>0,lastProgress=tic;end
    if remaining==0,fprintf('METRIC_LANE_COMPLETE %d\n',lane);return;end
    if progress==0,pause(60);end
end
fprintf('METRIC_LANE_IDLE_TIMEOUT %d; incomplete inputs remain\n',lane);
end

function h=sha256_file(p)
fid=fopen(p,'rb');assert(fid>=0);bytes=fread(fid,Inf,'*uint8');fclose(fid);
md=java.security.MessageDigest.getInstance('SHA-256');md.update(bytes);
h=lower(reshape(dec2hex(typecast(md.digest(),'uint8'),2).',1,[]));
end
