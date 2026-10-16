% addpath('MLRfuns');
% addpath('data_movie');
% addpath('evaluates');
% 
% tol1 = 1e-4;
% maxiter = 3000;
% lambda1=100;
% lambda2=1000;
% 
% ratios = 6;
% rsts_avg = zeros(ratios,5);
% rsts_std = zeros(ratios,5);
% partial = 0.1;
% [Right,Data,Left] = getdata("movie");
% Data = Data';
% Left = Left';
% Right = Right';
% 
% mid = 3;
% LL = getTransMat(size(Data,2), partial);
% 
% 
% for tstrat = 1:ratios
% sr = (tstrat*2-1)*0.1;
% if tstrat==6
%     sr=0.063;
% end
% tmp_times = 10;
% tmp_rst = zeros(tmp_times,5);
% for test = 1:tmp_times
% [posData, posTrain] = misObv2(Data, sr);
% posTest = posData-posTrain;
% tic;
% Tdata = (Data-mid).*posTrain;
% % output = MLR_APG(Tdata, Left, Right, lambda1, lambda2, maxiter, tol1, LL);
% output = MLR_trans(Tdata, 10000, LL);
% % output = MLR_noisy(Tdata, Left, Right, lambda1, lambda2, 0.1, maxiter, tol1, LL);
% % output = MLR_global(Tdata, Left, Right, 100, 100, maxiter, tol1, LL);
% output = output+mid;
% % dd = correct(Data,output,posTrain);
% % output = output + dd;
% time = toc;
% tmp_rst(test,:) = [evaluate_MAE(output,Data,posTest), evaluate_MRE(output,Data,0,posTest), evaluate_RMSE(output,Data,posTest), evaluate_RE(output,Data,posTest), time];
% end
% 
% rsts_avg(tstrat,:) = mean(tmp_rst);
% rsts_std(tstrat,:) = std(tmp_rst);
% end

% 
% addpath('MLRfuns');
% addpath('data_DTI');
% addpath('evaluates');
% tol1 = 0.01;
% maxiter = 1000;
% lambda1 = 10;
% lambda2 = 1;
% 
% numData = 2;
% ratios = 5;
% rsts_avg = zeros(numData*ratios,6);
% rsts_std = zeros(numData*ratios,6);
% for dataId = 1:numData
% [Left,Data,Right] = getdata("DTI",dataId);
% LL = getTransMat(size(Data,2));
% for tstrat = 1:ratios
% obvratio = (tstrat*2-1)/10;
% disp([dataId,tstrat]);
% 
% tmp_times = 10;
% tmp_rst = zeros(tmp_times,6);
% for test = 1:tmp_times
% obvT = genObv2(Data, obvratio);
% tic;
% M_incomp = (Data*2-1).*obvT;
% % output = MLR_APG(M_incomp, Left, Right, lambda1, lambda2, maxiter, tol1, LL);
% output = MLR_trans(M_incomp, 100, LL);
% % output = MLR_noisy(M_incomp, Left, Right, lambda1, lambda2, 10, maxiter, tol1, LL); %1
% % output = MLR_global(M_incomp, Left, Right, 1, 1, maxiter, tol1, LL);
% output = output/2+1/2;
% % dd = correct(Data,output,obvT);
% % output = output+dd;
% time = toc;
% loc = 1-obvT;
% tmp_rst(test,:) = [evaluate_MAE(output,Data,loc), evaluate_MRE(output,Data,0,loc), evaluate_RMSE(output,Data,loc), evaluate_RE(output,Data,loc), evaluate_AUC(output,Data,loc), time];
% end
% 
% rsts_avg((dataId-1)*ratios+tstrat,:) = mean(tmp_rst);
% rsts_std((dataId-1)*ratios+tstrat,:) = std(tmp_rst);
% end
% end
% 
% 
% 
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
numData = 2;
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
output = MLR_trans(M_incomp, 100, LL);
% output = MLR_APG(M_incomp, Left, Right, lambda1, lambda2, maxiter, tol, LL);
% output = MLR_noisy(M_incomp, Left, Right, lambda1, lambda2, 100, maxiter, tol, LL);
% output = MLR_global(M_incomp, Left, Right, 10, 10, maxiter, tol, LL);
output = output/2+1/2;
% dd = correct(Data,output,obvT);
% output = output+dd;
time = toc;
tmp_rst(test,:) = [evaluate_MLL(output,Data,0), time];
end

rsts_avg((dataId-1)*ratios+tstrat,:) = mean(tmp_rst);
rsts_std((dataId-1)*ratios+tstrat,:) = std(tmp_rst);
end
end

