function results = tune_cisf_parameters(data_or_path, target, split, options)
%   程序按 70%/10%/20% 划分训练集、验证集
%   和测试集。验证集 A-P 用于选择 ALPHA、GAMMA，验证集 H-L 用于选择
%   预测阈值，测试集只在最后评估一次。
%   beta=0.1、eta=100、tau=1.1，

    root_directory = fileparts(mfilename('fullpath'));
    addpath(fullfile(root_directory, 'metrics'));

    if nargin < 2
        target = [];
    end
    if nargin < 3
        split = [];
    end
    if nargin < 4 || isempty(options)
        options = struct();
    end

    if ischar(data_or_path) || (isstring(data_or_path) && isscalar(data_or_path))
        
        if isstruct(target) && isempty(split)
            split = target;
            target = [];
        end
        [data, target] = load_and_normalize_dataset(data_or_path);
    else
        data = data_or_path;
    end

    [data, target, n] = validate_data_and_target(data, target);
    if isempty(split)
        split = make_cisf_split(n);
    end
    validate_split(split, n);
    options = validate_options(options);

    alpha_values = options.alpha_values;
    gamma_values = options.gamma_values;
    validation_ap = zeros(numel(alpha_values), numel(gamma_values));
    best_ap = -inf;
    best_model = struct();

    train_data = subset_views(data, split.train);
    validation_data = subset_views(data, split.validation);
    validation_target = target(:, split.validation);

    for alpha_index = 1:numel(alpha_values)
        for gamma_index = 1:numel(gamma_values)
            para = default_cisf_parameters(numel(data));
            para.alpha = alpha_values(alpha_index);
            para.beta = 0.1;
            para.gamma = gamma_values(gamma_index);
            para.eta = 100;
            para.tau = 1.1;
            para.lambda = max(1, 12 * para.alpha) * ones(1, numel(data));
            para.max_iter = options.max_iter;
            para.tol = options.tol;
            para.ablation = options.ablation;

            
            evalc('[W_C, W_D, S, fitted_para] = train_L(train_data, target(:, split.train), para);');
            validation_scores = test_L(validation_data, W_C, W_D, S, fitted_para);
            ap = Average_precision(validation_scores, validation_target);
            if ~isfinite(ap)
                error('CISF:InvalidValidationSet', ...
                    'Validation Average Precision is undefined for this split.');
            end
            validation_ap(alpha_index, gamma_index) = ap;

            
            if ap > best_ap
                best_ap = ap;
                best_model.W_C = W_C;
                best_model.W_D = W_D;
                best_model.S = S;
                best_model.para = fitted_para;
                best_model.alpha = para.alpha;
                best_model.gamma = para.gamma;
                best_model.validation_scores = validation_scores;
            end
        end
    end

    [threshold, validation_hamming, threshold_candidates] = ...
        choose_validation_threshold(best_model.validation_scores, validation_target);
    best_model.para.thr = threshold;

    
    test_data = subset_views(data, split.test);
    test_target = target(:, split.test);
    test_scores = test_L(test_data, best_model.W_C, best_model.W_D, ...
        best_model.S, best_model.para);
    predicted_labels = double(test_scores > threshold);

    results.split = split;
    results.search.alpha_values = alpha_values;
    results.search.gamma_values = gamma_values;
    results.validation.average_precision = validation_ap;
    results.validation.best_average_precision = best_ap;
    results.validation.threshold = threshold;
    results.validation.hamming_loss = validation_hamming;
    results.validation.threshold_candidates = threshold_candidates;
    results.best.alpha = best_model.alpha;
    results.best.beta = 0.1;
    results.best.gamma = best_model.gamma;
    results.best.eta = 100;
    results.best.tau = 1.1;
    results.best.lambda0 = max(1, 12 * best_model.alpha);
    results.best.ablation = options.ablation;
    results.best.parameters = best_model.para;
    results.test.indices = split.test;
    results.test.scores = test_scores;
    results.test.predicted_labels = predicted_labels;
    results.test.metrics.HammingLoss = Hamming_loss(predicted_labels, test_target);
    results.test.metrics.RankingLoss = Ranking_loss(test_scores, test_target);
    results.test.metrics.OneError = One_error(test_scores, test_target);
    results.test.metrics.Coverage = coverage(test_scores, test_target);
    results.test.metrics.AveragePrecision = Average_precision(test_scores, test_target);
end

function [data, target] = load_and_normalize_dataset(dataset_path)
    if ~isfile(dataset_path)
        error('CISF:DatasetNotFound', 'Dataset file not found: %s', dataset_path);
    end
    loaded = load(dataset_path, 'data', 'target');
    if ~isfield(loaded, 'data') || ~isfield(loaded, 'target') || ~iscell(loaded.data)
        error('CISF:InvalidDataset', ...
            'The dataset must contain a cell array DATA and a TARGET matrix.');
    end
    [raw_data, target, n] = validate_data_and_target(loaded.data, loaded.target);
    data = cell(1, numel(raw_data));
    for view = 1:numel(raw_data)
        data{view} = normalization(raw_data{view}', 'l2', 1)';
        if size(data{view}, 2) ~= n
            error('CISF:InvalidDataset', 'Normalization changed the sample count.');
        end
    end
end

function [data, target, n] = validate_data_and_target(data, target)
    if ~iscell(data) || isempty(data)
        error('CISF:InvalidDataset', 'DATA must be a nonempty cell array of views.');
    end
    if ~isnumeric(target) || ~isreal(target) || ~ismatrix(target) || ...
            ~all(isfinite(target(:))) || ~all(target(:) == 0 | target(:) == 1)
        error('CISF:InvalidTarget', ...
            'TARGET must be a real, finite matrix using only binary labels {0,1}.');
    end
    n = size(target, 2);
    if n < 10
        error('CISF:SplitTooSmall', 'At least 10 samples are required for a 70/10/20 split.');
    end
    for view = 1:numel(data)
        if ~isnumeric(data{view}) || ~isreal(data{view}) || ~ismatrix(data{view}) || ...
                isempty(data{view}) || ~all(isfinite(data{view}(:))) || size(data{view}, 2) ~= n
            error('CISF:InvalidDataset', ...
                'Every DATA view must be a finite d-by-n matrix matching TARGET.');
        end
    end
end

function validate_split(split, n)
    required_fields = {'train', 'validation', 'test'};
    if ~isstruct(split) || ~all(isfield(split, required_fields))
        error('CISF:InvalidSplit', ...
            'SPLIT must contain train, validation, and test index vectors.');
    end
    all_indices = [];
    for field_index = 1:numel(required_fields)
        indices = split.(required_fields{field_index});
        if ~isnumeric(indices) || ~isvector(indices) || isempty(indices) || ...
                any(~isfinite(indices)) || any(indices ~= floor(indices)) || ...
                any(indices < 1) || any(indices > n)
            error('CISF:InvalidSplit', 'Each split partition must contain valid sample indices.');
        end
        all_indices = [all_indices, reshape(indices, 1, [])]; %#ok<AGROW>
    end
    if numel(all_indices) ~= n || numel(unique(all_indices)) ~= n
        error('CISF:InvalidSplit', ...
            'Split partitions must be disjoint and contain every sample exactly once.');
    end
end

function options = validate_options(options)
    if ~isstruct(options) || ~isscalar(options)
        error('CISF:InvalidOptions', 'OPTIONS must be a scalar struct.');
    end
    options = set_default_option(options, 'alpha_values', [1e-3, 1e-2, 1e-1, 1, 10, 100]);
    options = set_default_option(options, 'gamma_values', [0.01, 0.05, 0.1, 0.5, 1, 5, 10]);
    options = set_default_option(options, 'max_iter', 50);
    options = set_default_option(options, 'tol', 1e-3);
    options = set_default_option(options, 'ablation', 'full');
    validateattributes(options.alpha_values, {'numeric'}, {'vector', 'real', 'finite', 'positive'}, ...
        mfilename, 'options.alpha_values');
    validateattributes(options.gamma_values, {'numeric'}, {'vector', 'real', 'finite', 'positive'}, ...
        mfilename, 'options.gamma_values');
    validateattributes(options.max_iter, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'options.max_iter');
    validateattributes(options.tol, {'numeric'}, {'scalar', 'real', 'finite', 'nonnegative'}, ...
        mfilename, 'options.tol');
    options.ablation = cisf_ablation_mode(struct('ablation', options.ablation));
    options.alpha_values = reshape(options.alpha_values, 1, []);
    options.gamma_values = reshape(options.gamma_values, 1, []);
end

function options = set_default_option(options, name, value)
    if ~isfield(options, name) || isempty(options.(name))
        options.(name) = value;
    end
end

function data_subset = subset_views(data, indices)
    data_subset = cell(1, numel(data));
    for view = 1:numel(data)
        data_subset{view} = data{view}(:, indices);
    end
end

function [threshold, best_hamming, candidates] = choose_validation_threshold(scores, target)
    candidates = unique(scores(:))';
    if isempty(candidates) || any(~isfinite(candidates))
        error('CISF:InvalidValidationScores', ...
            'Validation scores must contain at least one finite value.');
    end
    best_hamming = inf;
    threshold = candidates(1);
    for candidate_index = 1:numel(candidates)
        candidate = candidates(candidate_index);
        loss = Hamming_loss(double(scores > candidate), target);
        if loss < best_hamming
            best_hamming = loss;
            threshold = candidate;
        end
    end
end
