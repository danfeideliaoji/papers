function [W, Xhat] = CPSO_TAL(X, c, options)
%% Input
    % X = cell(1, V) multi-view dataset, each cell is a view, each column represents a sample
    % options: struct with parameters (gamma, lambda, eta, r, k)
    % c: the number of clusters
%% Output
    % W: Feature selection matrix
    % Xhat: the imputed multi-view data

options.r = c; % Set the number of anchors to the number of clusters c
options.k = c; % Set the dimensionality of the low-dimensional subspace to the number of clusters c
V = size(X, 2); % Get the number of views V
[~, n] = size(X{1}); % Get the number of samples n
options.use_cpso = true;

% ========== Direct CPSO Optimization of W ==========
if isfield(options, 'use_cpso') && options.use_cpso
    % 1. Calculate the total parameter dimension of W
    d_list = zeros(1, V);
    w_dim = 0;
    for v = 1:V
        d_list(v) = size(X{v}, 1);
        w_dim = w_dim + d_list(v) * options.k;
    end
    num_particles = 21; % Conveniently divide into three equal parts
    max_cpso_iter = 20;
    swarm = rand(num_particles, w_dim); % Initialize the particle swarm
    velocity = zeros(num_particles, w_dim);
    pbest = swarm;
    pbest_fitness = inf(num_particles, 1);
    w = 0.7; c1 = 1.5; c2 = 1.5;
    K = floor(num_particles / 3); % Number of particles in each sub-group
    % Triple-system CPSO main loop
    for iter = 1:max_cpso_iter
        % 1. Fitness evaluation
        fitvals = inf(num_particles, 1);
        parfor i = 1:num_particles
            w_vec = swarm(i, :); % Current particle's W vector
            % Restore w_vec to cell format W
            W_cell = cell(1, V);
            cnt = 0;
            for v = 1:V
                d = d_list(v);
                W_cell{v} = reshape(w_vec(cnt + 1:cnt + d * options.k), d, options.k);
                cnt = cnt + d * options.k;
            end
            try
                [~, ~, fitval] = TAL(X, c, options, W_cell);
            catch
                fitval = inf;
            end
            fitvals(i) = fitval;
            if fitval < pbest_fitness(i)
                pbest_fitness(i) = fitval;
                pbest(i, :) = w_vec;
            end
        end
        % 2. Fitness sorting and clustering
        [~, idx] = sort(pbest_fitness);
        idx_baoc = idx(1:K);      % Retained group
        idx_huif = idx(K + 1:2*K);  % Recovered group
        idx_buyu = idx(2*K + 1:end); % Sterile group
        % 3. Independent CPSO for each sub-group
        % Retained group
        gbest_baoc = pbest(idx_baoc(1), :); % Global best of the retained group
        for j = 1:K
            i = idx_baoc(j);
            velocity(i, :) = w * velocity(i, :) + c1 * rand(1, w_dim) .* (pbest(i, :) - swarm(i, :)) + c2 * rand(1, w_dim) .* (gbest_baoc - swarm(i, :));
            swarm(i, :) = swarm(i, :) + velocity(i, :);
        end
        % Recovered group
        gbest_huif = pbest(idx_huif(1), :); % Global best of the recovered group
        for j = 1:K
            i = idx_huif(j);
            velocity(i, :) = w * velocity(i, :) + c1 * rand(1, w_dim) .* (pbest(i, :) - swarm(i, :)) + c2 * rand(1, w_dim) .* (gbest_huif - swarm(i, :));
            swarm(i, :) = swarm(i, :) + velocity(i, :);
        end
        % Sterile group
        gbest_buyu = pbest(idx_buyu(1), :); % Global best of the sterile group
        for j = 1:(num_particles - 2 * K)
            i = idx_buyu(j);
            velocity(i, :) = w * velocity(i, :) + c1 * rand(1, w_dim) .* (pbest(i, :) - swarm(i, :)) + c2 * rand(1, w_dim) .* (gbest_buyu - swarm(i, :));
            swarm(i, :) = swarm(i, :) + velocity(i, :);
        end
        % 4. Triple-system co-evolution (crossover + mutation + replacement)
        % Crossover between retained and recovered groups, generating new individuals
        num_cross = min(length(idx_buyu), K); % Number of crossovers
        for j = 1:num_cross
            p1 = swarm(idx_baoc(randi(K)), :);
            p2 = swarm(idx_huif(randi(K)), :);
            alpha = rand();
            child = alpha * p1 + (1 - alpha) * p2; % Linear crossover
            % Add Gaussian mutation
            child = child + 0.05 * randn(1, w_dim);
            % Replace the worst individual in the sterile group
            swarm(idx_buyu(j), :) = child;
            velocity(idx_buyu(j), :) = 0;
        end
        % 5. Global best
        [gbest_fitness, gbest_idx] = min(pbest_fitness);
        gbest = pbest(gbest_idx, :);
    end
    % ========== Run TIME_FS_core once with optimal W ==========
    w_vec = gbest;
    W = cell(1, V);
    cnt = 0;
    for v = 1:V
        d = d_list(v);
        W{v} = reshape(w_vec(cnt + 1:cnt + d * options.k), d, options.k);
        cnt = cnt + d * options.k;
    end
    [W, Xhat, ~] = TAL(X, c, options, W);
    return;
end
end


% Supporting external given W for TIME_FS_core
function [W, Xhat, objval] = TAL(X, c, options, W)
% Only use the given W to optimize parameters such as Z, A, M, P, etc.
V = size(X, 2);
[~, n] = size(X{1});

% ... (reuse the initialization and optimization process of TIME_FS_core, 
% just not updating W) ...

% -------------- Initialize E -------------- %
G = cell(1, V); E = cell(1, V); Xhat = cell(1, V);
for v = 1:V
    [dv, n_local] = size(X{v});
    idx_v = find(ismissing(X{v}(1, :)));
    ns = numel(idx_v);
    G{v} = sparse(1:ns, idx_v, 1, ns, n_local);
    Xv = X{v};
    Xv(:, idx_v) = 0;
    if n_local - ns > 0
        E{v} = (sum(Xv, 2) / (n_local - ns)) * ones(1, ns);
    else
        E{v} = zeros(dv, ns);
    end
    X{v} = Xv;
    Xhat{v} = Xv + E{v} * G{v};
end

% -------------- Initialize Z -------------- %
Z = zeros(options.k, n, V);
for v = 1:V
    tmp = rand(n, options.k);
    [zr, ~] = qr(tmp, 0);
    zr = zr(:, 1:options.k);
    Z(:, :, v) = zr';
end

% -------------- Initialize A M P -------------- %
A = rand(options.k, options.r);
P = rand(V, options.r);
M = updateM(Z, A, P, options);

% -------------- Initialize QW -------------- %
QW = cell(1, V);
for v = 1:V
    QW{v} = ones(size(W{v}));
end
alpha = updateAlpha(Xhat, W, Z, QW, options);

% -------------- Optimization -------------- %
max_iter = 50;
for iter = 1:max_iter
    Z = updateZ(Xhat, W, alpha, A, M, P, options);
    A = updateA(Z, M, P);
    P = updateP(Z, A, M);
    M = updateM(Z, A, P, options);
    E = updateE(W, Z, G);
    for v = 1:V
       Xhat{v} = X{v} + E{v} * G{v};
    end
    % QW optional: if dynamic updating is needed, reassign, otherwise keep unchanged
    % for v = 1:V
    %     QW{v} = ones(size(W{v}));
    % end
    alpha = updateAlpha(Xhat, W, Z, QW, options);
    objval = obj_fun(Xhat, W, Z, QW, alpha, A, M, P, options);
    if iter > 1 && abs(objval - objval(iter - 1)) < 1e-3
        break
    end
end
end

function f = obj_fun(Xhat, W, Z, QW, alpha, A, M, P, options)
%% Input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views.
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% alpha: Adaptive weights for each view
% A: Consensus anchor matrix
% M: Consensus anchor graph
% P: Anchor preference weight matrix
% options: Parameters 

%% Output
% f: objective function value

V = size(Xhat, 2); % Number of views

f1 = 0;
f2 = 0;
f3 = 0;
for v = 1:V
    Xv_hat = Xhat{v}; 
    f1 = f1 + (alpha(v)^options.gamma) * (norm(Xv_hat - W{v} * Z(:,:,v), 'fro')^2);  % Reconstruction error term
    f2 = f2 + (alpha(v)^options.gamma) * trace((W{v} .* QW{v})' * W{v}); % Sparsity regularization term
    f3 = f3 + norm(Z(:,:,v) - A * diag(P(v,:)) * M', 'fro')^2; % Low-dimensional consistency term
end

f4 = norm(M, 'fro')^2; % Anchor graph regularization term

f = f1 + options.lambda * f2 + f3 + options.eta * f4; % Total objective function
clear f1 f2 f3 f4
end


function A = updateA(Z, M, P)
%% Input 
% Z: Tensor formed by stacking low-dimensional representations of views
% M: Consensus anchor graph
% P: View preference weight matrix

%% Output
% A: Consensus anchor matrix

% Obtain the mode-1 unfolding of tensor Z
tensor_size = size(Z);
Z_mode_1 = reshape(Z, tensor_size(1), []); % Mode-1 unfolding
% Calculate P*M, where * is the Khatri-Rao product
PM = khatrirao(P, M); % Calculate Khatri-Rao product
% Update A
A = Z_mode_1 * PM / (PM' * PM); % Update A
clear tensor_size Z_mode_1 PM
end

function [M] = updateM(Z, A, P, options)
%% Input 
% Z: Tensor formed by stacking low-dimensional representations of views
% A: Consensus anchor matrix
% P: View preference weight matrix
% options: Parameters 

%% Output
% M: Consensus anchor graph

tensor_size = size(Z);
n = tensor_size(2);

% Obtain the horizontal mode-2 unfolding of tensor Z
Z_mode_2 = reshape(permute(Z, [2 1 3]), tensor_size(2), []); % Mode-2 unfolding
I = eye(options.r); % r is the number of anchors

% Calculate P*A, where * is the Khatri-Rao product
PA = khatrirao(P, A); % Calculate Khatri-Rao product

Y = -1 * Z_mode_2 * (PA) / (options.eta * I + (PA') * PA); % Calculate Y

% Update M
v1 = ones(options.r, 1); 
tmp = ((1 + Y * v1) / options.r) .* ones(n, options.r) - Y; % Calculate intermediate variables for M
M = max(tmp, 0); % Take the non-negative part
clear tensor_size Z_mode_2 PA Y tmp
end

function P = updateP(Z, A, M)
%% Input 
% Z: Tensor formed by stacking low-dimensional representations of views
% A: Consensus anchor matrix
% M: Consensus anchor graph
%% Output
% P: View preference weight matrix

% Obtain the horizontal mode-3 unfolding of tensor Z
tensor_size = size(Z);
Z_mode_3 = reshape(permute(Z, [3 1 2]), tensor_size(3), []); % Mode-3 unfolding

% Calculate P*M, where * is the Khatri-Rao product
MA = khatrirao(M, A); % Calculate Khatri-Rao product
% Update P
P = Z_mode_3 * MA / (MA' * MA); % Update P
clear Z_mode_3 tensor_size MA
end 

function [alpha] = updateAlpha(Xhat, W, Z, QW, options)
%% Input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% options: parameters 

%% Output
% alpha: Adaptive weights of each view

V = size(Xhat, 2); 

d = zeros(1, V); 
for v = 1:V
   d(v) = norm(Xhat{v} - W{v} * Z(:,:,v), 'fro')^2 + ...
        options.lambda * trace((W{v} .* QW{v})' * W{v});   % Calculate the loss for each view
end


% Update alpha
alpha = d.^(1/(1-options.gamma))/sum(d.^(1/(1-options.gamma))); % 更新权重
end

function [E] = updateE(W,Z,G)
%% Input 
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views
% G: Missing samples indicator matrix for each view 
% options: Parameters 
%% output
% E: Adaptive imputation matrix for missing samples in each view

V = size(W,2); 

% Update E
E = cell(1,V);
for v =1:V
    E{v} = full(W{v}*(Z(:,:,v)*G{v}')); % 计算补全矩阵
end

end

function [W, QW] = updateW(Xhat, Z, QW, options)
%% Input 
% Xhat: The imputed multi-view dataset
% Z: Tensor formed by stacking low-dimensional representations of views
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% options: Parameters

%% Output
% W: Feature selection matrix
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W

V = size(Xhat, 2);
W = cell(1, V);

for v = 1:V
    Xv_hat = Xhat{v};
    W{v} = (Xv_hat * Z(:,:,v)') ./ (1 + options.lambda * QW{v}); % Update W
    QW{v} = 1 ./ (2 * sqrt(sum(W{v} .* W{v}, 2) + eps)); % Update QW
end

end

function [Z] = updateZ(Xhat, W, alpha, A, M, P, options)
%% Input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% alpha: Adaptive weights for each view
% A: Consensus anchor matrix
% M: Consensus anchor graph
% P: Anchor preference weight matrix
% options: Parameters 

%% Output
% Z: Tensor formed by stacking low-dimensional representations of views.

V = size(Xhat, 2); % Number of views

[~, n] = size(Xhat{1}); % Number of samples

% Update Z
Z = zeros(options.k, n, V); % Initialize Z
for v = 1:V
    Cv = A * diag(P(v, :)) * M'; % Compute consistency term
    Qv = (alpha(v)^options.gamma) * W{v}' * Xhat{v} + Cv; % Intermediate variable
    [Uv, ~, Vv] = svd(Qv, 'econ'); % Singular value decomposition
    Z(:,:,v) = Uv * Vv'; % Update Z
end

end


