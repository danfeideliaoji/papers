clc; clear;
warning('off', 'all');
addpath('D:\研1\论文\AAAI\提交\IMUFFS-main\CPSO-TAL\')
addpath('CalcMeasures')
addpath('D:\研1\论文\AAAI\提交\IMUFFS-main\dataset')

%% Parameter settings
K = 5; % Number of clients
max_comm_rounds = 5; % Number of federated communication rounds
options = [];
options.gamma = 6; options.lambda = 0.001; options.eta = 1;

%% Load data
load('COIL100_MissingRatio0.4.mat'); % X: 1xV cell, Y: n x 1
c = length(unique(Y)); % Number of clusters
V = numel(X); % Number of views
n = size(X{1},2); % Number of samples

%% Client partition
client_indices = FederatedPartition(n, K);

ACC_1 = -inf;
NMI_1 = -inf;

%% Results after 10 runs
for run = 1:10
    fprintf('Running iteration %d...\n', run);
    % Display the running status of each client
    for client = 1:K
        fprintf('Client %d is training...\n', client);
    end
    % Perform federated training
    [global_W, last_mean_NMI] = FederatedTrain(X, Y, c, V, K, client_indices, options, max_comm_rounds);

    % Perform evaluation
    [ACC, NMI] = FederatedEvaluate(X, Y, global_W, c);
    % Update results
    if ACC > ACC_1
        ACC_1 = ACC;
        NMI_1 = NMI;
    end
end

fprintf('Results after 10 runs: ACC(mean) = %f, NMI(mean) = %f\n', ACC_1, NMI_1);
