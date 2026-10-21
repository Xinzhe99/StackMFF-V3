function score=metricsFMI_wavelet_vectorized(A,B,F)
% Batched port of metricsFMI_wavelet.m (Haghighat FAST-FMI / MEFB).
% Retains its dmey features, 3x3 windows, branch order and Nelsen formulas.
% In particular the second equality test uses the fSub modified by the
% first comparison, exactly as the reference implementation does.
assert(ismatrix(A) && isequal(size(A),size(B),size(F)));
A=feature(A);B=feature(B);F=feature(F);
[nr,nc]=size(A);n=(nr-2)*(nc-2);values=zeros(1,n);
offset=reshape((0:2)'+(0:2)*nr,9,1);
for first=1:8192:n
    k=first:min(first+8191,n);[r,c]=ind2sub([nr-2,nc-2],k);
    ix=r+(c-1)*nr+offset;
    a=A(ix);b=B(ix);f=F(ix);
    eqA=all(a==f,1);fn=normalize(f);
    carried=fn;carried(:,eqA)=f(:,eqA);
    eqB=all(b==carried,1);
    af=pair_score(normalize(a),fn);af(eqA)=1;
    bf=pair_score(normalize(b),normalize(carried));bf(eqB)=1;
    values(k)=(af+bf)/2;
end
score=mean(values);
end

function X=feature(X)
[a,h,v,d]=dwt2(double(X),'dmey');X=[a,h;v,d];
lo=min(X(:));hi=max(X(:));
if hi==lo,X=ones(size(X));else,X=(X-lo)/(hi-lo);end
end

function N=normalize(X)
lo=min(X,[],1);delta=max(X,[],1)-lo;
constant=delta==0;delta(constant)=1;
N=(X-lo)./delta;N(:,constant)=1;
end

function score=pair_score(X,F)
% Match sum(sum(3x3 patch)), including its grouping of column sums.
X=X./(sum(X(1:3,:),1)+sum(X(4:6,:),1)+sum(X(7:9,:),1));
F=F./(sum(F(1:3,:),1)+sum(F(4:6,:),1)+sum(F(7:9,:),1));
xc=cumsum(X,1);fc=cumsum(F,1);
xt=X-mean(X,1);ft=F-mean(F,1);numer=sum(xt.*ft,1);
c=numer./sqrt(sum(xt.*xt,1).*sum(ft.*ft,1));c(numer==0)=0;
ex=zeros(1,size(X,2));ef=ex;ex2=ex;ef2=ex;
for i=1:9
    ex=ex+i*X(i,:);ex2=ex2+i^2*X(i,:);
    ef=ef+i*F(i,:);ef2=ef2+i^2*F(i,:);
end
xs=sqrt(ex2-ex.^2);fs=sqrt(ef2-ef.^2);
lower=c<0;covariance=zeros(size(c));
U=zeros(10,10,size(X,2));
for i=1:9
    for j=1:9
        s=fc(i,:)+xc(j,:);
        u=0.5*(s-abs(fc(i,:)-xc(j,:)));
        u(lower)=0.5*(s(lower)-1+abs(s(lower)-1));
        U(i+1,j+1,:)=u;
        covariance=covariance+u-fc(i,:).*xc(j,:);
    end
end
weight=zeros(size(c));sel=c~=0 & xs~=0 & fs~=0;
bound=covariance./(fs.*xs);weight(sel)=c(sel)./bound(sel);
% The reference accumulates (1,1), first column, first row, then interior.
order=[1,1;(2:9)',ones(8,1);ones(8,1),(2:9)'];
for i=2:9,for j=2:9,order(end+1,:)=[i,j];end,end %#ok<AGROW>
H=zeros(size(c));
for k=1:81
    i=order(k,1);j=order(k,2);
    joint=reshape(U(i+1,j+1,:)-U(i,j+1,:)-U(i+1,j,:)+U(i,j,:),1,[]);
    p=weight.*joint+(1-weight).*F(i,:).*X(j,:);
    nz=p~=0;term=zeros(size(p));term(nz)=real(-p(nz).*log2(p(nz)));
    H=H+term;
end
hx=entropy(X);hf=entropy(F);mi=hx+hf-H;
score=2*mi./(hx+hf);score(mi==0)=0;
end

function h=entropy(p)
q=p;q(q==0)=1;h=sum(-p.*log2(q),1);
end
