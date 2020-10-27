function eigvector = MFA(deepFeaTrn, trainlabelss, channel, no_dims, kw, kb)

%deepFeaTrn =single(deepFeaTrn);
%trainlabelss = single(trainlabelss);
%channel = single(channel);
%no_dims = single(no_dims);
%kw = single(kw);
%kb = single(kb);



[aa bb]=size(trainlabelss);
trainlabels = [];
for i = 1:aa
    trainlabels = [trainlabels, repmat(trainlabelss(i),1, channel)];
end
trainlabels = trainlabels';

k = kw;
t = k;
Reg = 0.001;
% no_dims = 30;
gnd = trainlabels;
% class_num = max(gnd);

[n m] = size(deepFeaTrn);
Sb=zeros(n);
Sw=zeros(n);


for i=1:n
    indb=find(gnd~=gnd(i));
    if kb >= length(indb)
        kb = length(indb);
    else
        kb = kb;
    end
    dd=EuDist2(deepFeaTrn(i,:),deepFeaTrn(indb,:));
    [~,ids]=sort(dd);
    Sb(i,indb(ids(1:kb))) = exp(-dd(ids(1:kb)).^2/(2*mean(dd(ids(1:kb)))^2));

    indw = find(gnd==gnd(i));
    if t >= length(indw)
        kw = length(indw)-1;
    else
        kw = t;
    end
    dd=EuDist2(deepFeaTrn(i,:),deepFeaTrn(indw,:));
    [~,ids]=sort(dd);
    Sw(i,indw(ids(2:kw+1))) = exp(-dd(ids(2:kw+1)).^2/(2*mean(dd(ids(2:kw+1)))^2));
end

Sb=max(Sb,Sb');
Sw=max(Sw,Sw');

Db=diag(sum(Sb,2));
Mb= (Db - Sb);

Dw=diag(sum(Sw,2));
Mw= (Dw - Sw);

Mb=max(Mb,Mb');
Mw=max(Mw,Mw');
A=deepFeaTrn'*Mb*deepFeaTrn;
B=deepFeaTrn'*Mw*deepFeaTrn;

A=A+eye(size(A,1))*Reg;% Regulation
B=B+eye(size(B,1))*Reg;

A=(A+A')/2;
B=(B+B')/2;

opts=[];opts.disp = 0;opts.isreal = 1; opts.issym = 1;
[eigvector, eigvalue]=eigs(A,B,no_dims,'la',opts);
for i = 1:size(eigvector,2)
    eigvector(:,i) = eigvector(:,i)./norm(eigvector(:,i));
end

% P1 = eigvector;
% 
% DMML_Train = deepFeaTrn*P1;
% % aaa = size(DMML_Train,1);
% % DMML_Train = [ones(aaa,1) DMML_Train];


