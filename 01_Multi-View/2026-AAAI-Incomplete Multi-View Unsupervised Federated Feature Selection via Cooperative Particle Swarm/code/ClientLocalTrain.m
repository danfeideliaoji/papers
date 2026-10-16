function [W_local, nmi] = ClientLocalTrain(X_local, Y_local, c, options)
[W_local, Xhat_local] = CPSO_TAL(X_local, c, options);
Xhat_mat = cell2mat(Xhat_local');
W_mat = cell2mat(W_local');
[~, idx] = sort(sum(W_mat.*W_mat,2), 'descend');
d = size(Xhat_mat,1);
fea_ratio = 0.4;
feaSet = Xhat_mat(idx(1:round(fea_ratio*d)), :)';
nan_mask = any(isnan(feaSet),2);
feaSet_clean = feaSet(~nan_mask,:);
Y_local_clean = Y_local(~nan_mask);
if numel(Y_local_clean) < c || numel(unique(Y_local_clean)) < 2
    nmi = 0;
    return;
end
yhat = kmeans(feaSet_clean, c, 'Start', 'plus', 'Replicates', 3);
yhat = bestMap(Y_local_clean, yhat);
nmi = MutualInfo(Y_local_clean, yhat);
end 