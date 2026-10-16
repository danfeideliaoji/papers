addpath('MLRfuns');
addpath('data_movie');
addpath('evaluates');

tol1 = 1e-4;
maxiter = 3000;
lambda1=100;
lambda2=1000;

ratios = 6;
rsts_avg = zeros(ratios,5);
rsts_std = zeros(ratios,5);
partial = 0.1;
[Right,Data,Left] = getdata("movie");
Data = Data';
Left = Left';
Right = Right';

mid = 3;
LL = getTransMat(size(Data,2), partial);


for tstrat = 1:ratios
sr = (tstrat*2-1)*0.1;
if tstrat==6
    sr=0.063;
end
tmp_times = 10;
tmp_rst = zeros(tmp_times,5);
for test = 1:tmp_times
[posData, posTrain] = misObv2(Data, sr);
posTest = posData-posTrain;
tic;
Tdata = (Data-mid).*posTrain;
output = MLR_APG(Tdata, Left, Right, lambda1, lambda2, maxiter, tol1, LL);
% output = MLR_trans(Tdata, 10000, LL);
% output = MLR_noisy(Tdata, Left, Right, lambda1, lambda2, 0.1, maxiter, tol1, LL);
% output = MLR_global(Tdata, Left, Right, 100, 100, maxiter, tol1, LL);
output = output+mid;
dd = correct(Data,output,posTrain);
output = output + dd;
time = toc;
tmp_rst(test,:) = [evaluate_MAE(output,Data,posTest), evaluate_MRE(output,Data,0,posTest), evaluate_RMSE(output,Data,posTest), evaluate_RE(output,Data,posTest), time];
end

rsts_avg(tstrat,:) = mean(tmp_rst);
rsts_std(tstrat,:) = std(tmp_rst);
end
