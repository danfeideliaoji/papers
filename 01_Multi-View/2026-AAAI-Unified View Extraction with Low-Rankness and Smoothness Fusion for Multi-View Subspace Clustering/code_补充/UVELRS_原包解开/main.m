%%
 
clear
clc
addpath ./ClusteringMeasure/
addpath ./twist
addpath ./differential_operators/
path = './data/';

%—— data ——
load(fullfile(path,'NGs.mat'));      

Y = truelabel{1};
gt   = Y;
X = data;
numClust = numel(unique(Y));            
% for i=1:6
%     X{i}=X{i}';
% end

lambdas1 = 0.05;
lambdas2 = 0.01;

% MSRCV  0.05 0.01
% NGs    0.01 0.01
% BBCSport 0.05 0.01 
% HW 0.05 0.01
nL       = numel(lambdas1);
ACC_mat  = zeros(nL,nL);


tic
for i = 1:nL
    for j = 1:nL
        %——  ——
        param.lambda1 = lambdas1(i);
        param.lambda2 = lambdas2(j);
        
        %—— ——
        Z = UVELRS(X, gt, param);
        S = 0;
        for k = 1:length(X)+1
            S = S + 0.5*(abs(Z{k}) + abs(Z{k})');
        end
        Cresults = SpectralClustering(S, numClust);
        ACC_mat(i,j) = Accuracy(Cresults, double(gt));
        
        fprintf('λ1=%.3g, λ2=%.3g  →  ACC=%.4f\n', lambdas1(i), lambdas2(j), ACC_mat(i,j));
    end
end
toc

%——  ——
[maxACC, idx] = max(ACC_mat(:));
[i_best, j_best] = ind2sub([nL,nL], idx);
best_lambda1 = lambdas1(i_best);
best_lambda2 = lambdas2(j_best);

fprintf('\n best ACC = %.4f， (lambda1, lambda2) = (%.3g, %.3g)\n', ...
        maxACC, best_lambda1, best_lambda2);
