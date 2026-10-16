addpath('MLRfuns');
addpath('data_MLL');
addpath('evaluates');
datasets = {'Arts','Business','Computers','Education','Health','Recreation','Society'};
data_path = 'data_MLL/';
tol = 0.01;
maxiter = 1000;
lambda1 = 10;
lambda2 = 10;

ratios = 5;
numData = 7;
rsts_avg = zeros(numData*ratios,12);
rsts_std = zeros(numData*ratios,12);
for dataId = 1:numData
    [Left,Data,Right] = getdata("MLL",dataId);
    LL = getTransMat(size(Data,2));
for tstrat = 1:ratios
obvratio = (tstrat*2-1)/10;
disp([dataId,tstrat]);

tmp_times = 10;
tmp_rst = zeros(tmp_times,12);
for test = 1:tmp_times
obvT = genObv2(Data,obvratio);
tic;
M_incomp = (Data*2-1).*obvT;
% output = MLR_APG(M_incomp, Left, Right, lambda1, lambda2, maxiter, tol, LL);
output = MLR_noisy(M_incomp, Left, Right, lambda1, lambda2, 100, maxiter, tol, LL);
% output = MLR_global(M_incomp, Left, Right, 10, 10, maxiter, tol, LL);
output = output/2+1/2;
dd = correct(Data,output,obvT);
output = output+dd;
time = toc;
tmp_rst(test,:) = [evaluate_MLL(output,Data,0), time];
end

rsts_avg((dataId-1)*ratios+tstrat,:) = mean(tmp_rst);
rsts_std((dataId-1)*ratios+tstrat,:) = std(tmp_rst);
end
end
