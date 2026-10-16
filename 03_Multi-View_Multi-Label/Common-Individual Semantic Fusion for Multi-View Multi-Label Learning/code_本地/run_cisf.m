function results = run_cisf(dataset_path, num_runs, seed)
%RUN_CISF Run the CISF experiment with independent 70/10/20 splits.
%   RESULTS = RUN_CISF(DATASET_PATH, NUM_RUNS, SEED) loads a MAT-file that
%   contains DATA (a cell array of d-by-n views) and TARGET (a real, finite
%   binary {0,1} label matrix by n).
%   Each run trains only on its training partition and evaluates the five
%   published metrics on the held-out test partition.  The validation
%   partition is retained in RESULTS for later parameter/threshold selection;
%   this baseline keeps the explicit default threshold in PARA.

    root_directory = fileparts(mfilename('fullpath'));
    if nargin < 1 || isempty(dataset_path)
        dataset_path = fullfile(root_directory, 'dataset', 'emotions.mat');
    end
    if nargin < 2 || isempty(num_runs)
        num_runs = 5;
    end
    if nargin < 3 || isempty(seed)
        seed = 1;
    end

    validateattributes(num_runs, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'num_runs');
    validateattributes(seed, {'numeric'}, {'scalar', 'integer', 'nonnegative'}, ...
        mfilename, 'seed');
    if ~(ischar(dataset_path) || (isstring(dataset_path) && isscalar(dataset_path)))
        error('CISF:InvalidDatasetPath', 'dataset_path must be a character vector or string scalar.');
    end
    if ~isfile(dataset_path)
        error('CISF:DatasetNotFound', 'Dataset file not found: %s', dataset_path);
    end

    addpath(fullfile(root_directory, 'metrics'));
    loaded = load(dataset_path, 'data', 'target');
    if ~isfield(loaded, 'data') || ~iscell(loaded.data) || ~isfield(loaded, 'target')
        error('CISF:InvalidDataset', ...
            'The dataset must contain a cell array DATA and a TARGET matrix.');
    end

    raw_data = loaded.data;
    target = loaded.target;
    if ~isnumeric(target) || ~isreal(target) || ~ismatrix(target) || ...
            ~all(isfinite(target(:))) || ~all(target(:) == 0 | target(:) == 1)
        error('CISF:InvalidTarget', ...
            'TARGET must be a real, finite matrix using only binary labels {0,1}.');
    end
    v_num = numel(raw_data);
    if v_num == 0
        error('CISF:InvalidDataset', 'DATA must contain at least one view.');
    end
    n = size(target, 2);
    if n < 10
        error('CISF:InvalidDataset', 'TARGET must contain at least 10 samples.');
    end

    data = cell(1, v_num);
    for view = 1:v_num
        if ~ismatrix(raw_data{view}) || size(raw_data{view}, 2) ~= n
            error('CISF:InvalidDataset', ...
                'Every DATA view must be a d-by-n matrix with n equal to size(TARGET, 2).');
        end
        % Normalize each sample and retain the original bias feature.
        data{view} = normalization(raw_data{view}', 'l2', 1)';
    end

    rng(seed, 'twister');
    metric_names = {'HammingLoss', 'RankingLoss', 'OneError', 'Coverage', 'AveragePrecision'};
    values = zeros(num_runs, numel(metric_names));
    runs = repmat(struct('split', [], 'metrics', [], 'parameters', []), 1, num_runs);

    for run_index = 1:num_runs
        split = make_cisf_split(n);
        train_data = cell(1, v_num);
        test_data = cell(1, v_num);
        for view = 1:v_num
            train_data{view} = data{view}(:, split.train);
            test_data{view} = data{view}(:, split.test);
        end

        para = default_cisf_parameters(v_num);
        [W_C, W_D, S, para] = train_L(train_data, target(:, split.train), para);
        prediction = test_L(test_data, W_C, W_D, S, para);
        predicted_labels = double(prediction > para.thr);
        test_target = target(:, split.test);

        values(run_index, :) = [ ...
            Hamming_loss(predicted_labels, test_target), ...
            Ranking_loss(prediction, test_target), ...
            One_error(prediction, test_target), ...
            coverage(prediction, test_target), ...
            Average_precision(prediction, test_target)];

        runs(run_index).split = split;
        runs(run_index).metrics = values(run_index, :);
        runs(run_index).parameters = para;
    end

    results.metric_names = metric_names;
    results.values = values;
    results.runs = runs;
    results.summary.mean = mean(values, 1);
    results.summary.std = std(values, 0, 1);
end
