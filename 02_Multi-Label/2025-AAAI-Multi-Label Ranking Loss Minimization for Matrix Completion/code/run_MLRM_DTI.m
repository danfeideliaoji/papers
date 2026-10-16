addpath('MLRfuns');
addpath('data_DTI');
addpath('evaluates');
tol1 = 0.01;
maxiter = 1000;
lambda1 = 10;
lambda2 = 1;

numData = 2;
ratios = 5;
rsts_avg = zeros(numData*ratios,6);
rsts_std = zeros(numData*ratios,6);
for dataId = 1:numData
[Left,Data,Right] = getdata("DTI",dataId);
LL = getTransMat(size(Data,2));
for tstrat = 1:ratios
obvratio = (tstrat*2-1)/10;
disp([dataId,tstrat]);

tmp_times = 10;
tmp_rst = zeros(tmp_times,6);
for test = 1:tmp_times
obvT = genObv2(Data, obvratio);
tic;
M_incomp = (Data*2-1).*obvT;
output = MLR_APG(M_incomp, Left, Right, lambda1, lambda2, maxiter, tol1, LL);
% output = MLR_trans(M_incomp, 100, LL);
% output = MLR_noisy(M_incomp, Left, Right, lambda1, lambda2, 10, maxiter, tol1, LL); %1
% output = MLR_global(M_incomp, Left, Right, 1, 1, maxiter, tol1, LL);
output = output/2+1/2;
dd = correct(Data,output,obvT);
output = output+dd;
time = toc;
loc = 1-obvT;
tmp_rst(test,:) = [evaluate_MAE(output,Data,loc), evaluate_MRE(output,Data,0,loc), evaluate_RMSE(output,Data,loc), evaluate_RE(output,Data,loc), evaluate_AUC(output,Data,loc), time];
end

rsts_avg((dataId-1)*ratios+tstrat,:) = mean(tmp_rst);
rsts_std((dataId-1)*ratios+tstrat,:) = std(tmp_rst);
end
end
