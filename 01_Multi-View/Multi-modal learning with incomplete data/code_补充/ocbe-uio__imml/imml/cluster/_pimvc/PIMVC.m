function [Y,obj] = PIMVC(X,dim,ind_folds,lambda,beta,max_iter,options)
normX = 0;
linshi_GG = 0;
linshi_LS = 0;
for iv = 1:length(X)
    ind_1 = find(ind_folds(:,iv) == 1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X{iv}(:,ind_0) = [];
    linshi_W = diag(ind_folds(:,iv));
    linshi_W(:,ind_0) = [];
    G{iv} = linshi_W;
    linshi_St = X{iv}*X{iv}'+lambda*eye(size(X{iv},1));
    St2{iv} = mpower(linshi_St,-0.5);
    St3{iv} = St2{iv}*X{iv}*G{iv}';
    linshi_GG = linshi_GG+ind_folds(:,iv);
    linshi_W = full(constructW(X{iv}',options));
    W_graph = (linshi_W+linshi_W')*0.5;
    W_graph = G{iv}*W_graph*G{iv}';
    Sum_S = sum(W_graph);
    linshi_LS = linshi_LS+diag(Sum_S)-W_graph;
    options_dim = [];
    options_dim.ReducedDim = dim;
    [P1,~] = PCA1(X{iv}', options_dim);
    Piv{iv} = P1';
    XG{iv} = X{iv}*G{iv}';
    normX = normX+norm(X{iv},'fro')^2;
end
numInst  = size(G{1},1); 
Y = rand(dim,numInst);
inv_GS = inv(linshi_LS*beta+diag(linshi_GG));

for iter = 1:max_iter
    % ------------- Y ------------- %
    linshi_H1 = 0;
    for iv = 1:length(X)
        linshi_H1 = linshi_H1 + Piv{iv}*XG{iv};
    end
    Y = linshi_H1*inv_GS;
    
    % ----------------- Piv ------------------- %
    for iv = 1:length(X)
        linshi_M = St3{iv}*Y';
        linshi_M(isnan(linshi_M)) = 0;
        linshi_M(isinf(linshi_M)) = 0;
        [linshi_U,~,linshi_V] = svd(linshi_M','econ');
        linshi_U(isnan(linshi_U)) = 0;
        linshi_U(isinf(linshi_U)) = 0;
        linshi_V(isnan(linshi_V)) = 0;
        linshi_V(isinf(linshi_V)) = 0;        
        Piv{iv} = linshi_U*linshi_V'*St2{iv};
    end

    % -------------- obj --------------- %
    linshi_obj = 0;
    for iv = 1:length(X)
        linshi_obj = linshi_obj+norm(Piv{iv}*X{iv}-Y*G{iv},'fro')^2+lambda*norm(Piv{iv},'fro')^2;        
    end
    obj(iter) = (linshi_obj+beta*trace(Y*linshi_LS*Y'))/normX;
    if iter >3 && abs(obj(iter)-obj(iter-1))<1e-5
        %iter
        break;
    end
end
end