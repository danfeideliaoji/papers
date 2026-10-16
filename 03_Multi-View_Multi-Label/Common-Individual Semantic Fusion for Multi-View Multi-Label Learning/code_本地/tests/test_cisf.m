function tests = test_cisf
% Regression tests for the public CISF setup utilities.
tests = functiontests(localfunctions);
end

function setupOnce(~)
% Make the package functions visible when MATLAB runs this test by file path.
testDirectory = fileparts(mfilename('fullpath'));
addpath(fileparts(testDirectory));
end

function testDefaultParametersHaveRequiredFields(testCase)
% The experiment entry point must be able to obtain every training setting.
p = default_cisf_parameters(2);
verifyTrue(testCase, all(isfield(p, ...
    {'alpha','beta','gamma','eta','mu','lambda','lambda_max','tau','thr','max_iter'})));
end

function testNormalizationSupportsRectangularInput(testCase)
% Column-wise normalisation must preserve an n-by-d input when n differs from d.
x = [1 2; 3 4; 5 6];
verifySize(testCase, normalization(x, 'minmax', 0), size(x));
verifySize(testCase, normalization(x, 'meanstd', 0), size(x));
end

function testRunCisfRejectsNonBinaryTargets(testCase)
% Labels must retain the {0,1} encoding expected by training and metrics.
dataset_path = [tempname, '.mat'];
cleanup = onCleanup(@() delete_if_present(dataset_path)); %#ok<NASGU>
data = {rand(2, 10)};
target = [0.5, zeros(1, 9)];
save(dataset_path, 'data', 'target');

verifyError(testCase, @() run_cisf(dataset_path, 1, 1), 'CISF:InvalidTarget');
end

function testUpdateSUsesEveryView(testCase)
% The consensus label map must respond to every input view, not just the last.
p = default_cisf_parameters(2);
p.mu = [1, 1];
p.lambda = [1, 1];

Y = [1, 0, 1, 0; 0, 1, 0, 1];
X = {[1, 0, 1, 0; 0, 1, 0, 1], [1, 1, 0, 0; 0, 0, 1, 1]};
W_C = {eye(2), eye(2)};
W_D = {zeros(2), zeros(2)};
C = {Y, Y};
M = {zeros(2, 4), zeros(2, 4)};

baseline = Update_S(X, Y, W_C, W_D, C, M, p);
W_C{1} = 3 * eye(2);
changed_first_view = Update_S(X, Y, W_C, W_D, C, M, p);

verifyGreaterThan(testCase, norm(baseline - changed_first_view, 'fro'), 1e-8);
verifyTrue(testCase, all(isfinite(baseline(:))));
verifyEqual(testCase, diag(baseline), ones(size(Y, 1), 1), 'AbsTol', 1e-12);
end

function testTrainLProducesFiniteSmallModel(testCase)
% A small two-view training problem should finish with finite model outputs.
p = default_cisf_parameters(2);
p.max_iter = 2;
p.tol = 0;
p.eta = 1e-2;
p.beta = 1e-2;
p.gamma = 1e-3;
p.alpha = 1e-3;
Y = [1, 0, 1, 0, 1, 0; 0, 1, 0, 1, 0, 1];
X = {[1, 0, 1, 0, 1, 0; 0, 1, 0, 1, 0, 1; 1, 1, 1, 1, 1, 1], ...
     [0, 1, 0, 1, 0, 1; 1, 0, 1, 0, 1, 0; 1, 1, 1, 1, 1, 1]};

[W_C, W_D, S, out_p] = train_L(X, Y, p);
scores = test_L(X, W_C, W_D, S, out_p);

verifyTrue(testCase, all(isfinite(S(:))));
verifyTrue(testCase, all(cellfun(@(w) all(isfinite(w(:))), W_C)));
verifyTrue(testCase, all(cellfun(@(w) all(isfinite(w(:))), W_D)));
verifyTrue(testCase, all(cellfun(@(w) all(w(:) >= -1e-12), W_D)));
verifyTrue(testCase, all(isfinite(scores(:))));
verifyEqual(testCase, sum(out_p.mu), 1, 'AbsTol', 1e-12);
end

function testAblationsDisableTheRequestedPredictionComponent(testCase)
% The public ablation flag must remove the selected term from prediction.
p = default_cisf_parameters(1);
X = {[1, 2; 3, 4]};
W_C = {[2, 0; 0, 2]};
W_D = {[0.5, 0; 0, 0.5]};
S = [1, 3; 2, 1];

p.ablation = 'no_common';
scores_without_common = test_L(X, W_C, {zeros(2)}, S, p);
verifyEqual(testCase, scores_without_common, zeros(2), 'AbsTol', 1e-12);

p.ablation = 'no_individual';
scores_without_individual = test_L(X, {zeros(2)}, W_D, S, p);
verifyEqual(testCase, scores_without_individual, zeros(2), 'AbsTol', 1e-12);

p.ablation = 'no_label_correlation';
scores_without_label_correlation = test_L(X, W_C, {zeros(2)}, S, p);
verifyEqual(testCase, scores_without_label_correlation, W_C{1} * X{1}, 'AbsTol', 1e-12);
end

function testTrainLAblationsPreserveDisabledComponents(testCase)
% Each paper ablation must remain finite while enforcing its removal.
p = small_test_parameters();
Y = [1, 0, 1, 0, 1, 0; 0, 1, 0, 1, 0, 1];
X = {[1, 0, 1, 0, 1, 0; 0, 1, 0, 1, 0, 1; 1, 1, 1, 1, 1, 1], ...
     [0, 1, 0, 1, 0, 1; 1, 0, 1, 0, 1, 0; 1, 1, 1, 1, 1, 1]};

p.ablation = 'no_common';
[W_C, W_D, S, fitted] = train_L(X, Y, p);
verifyTrue(testCase, all(cellfun(@(w) all(abs(w(:)) < 1e-12), W_C)));
verifyTrue(testCase, all(isfinite(test_L(X, W_C, W_D, S, fitted)), 'all'));

p.ablation = 'no_individual';
[W_C, W_D, S, fitted] = train_L(X, Y, p);
verifyTrue(testCase, all(cellfun(@(w) all(abs(w(:)) < 1e-12), W_D)));
verifyTrue(testCase, all(isfinite(test_L(X, W_C, W_D, S, fitted)), 'all'));

p.ablation = 'no_label_correlation';
[W_C, W_D, S, fitted] = train_L(X, Y, p);
verifyEqual(testCase, S, eye(size(Y, 1)), 'AbsTol', 1e-12);
verifyTrue(testCase, all(isfinite(test_L(X, W_C, W_D, S, fitted)), 'all'));
end

function testPaperParameterMappingsArePresent(testCase)
% Pin the paper-to-code parameter mapping in the core update routines.
root = fileparts(fileparts(mfilename('fullpath')));
update_c = fileread(fullfile(root, 'Update_C.m'));
update_wc = fileread(fullfile(root, 'Update_WC.m'));
update_wd = fileread(fullfile(root, 'Update_WD.m'));

verifyTrue(testCase, contains(update_c, 'para.mu(j)'));
verifyTrue(testCase, contains(update_c, 'hsic_scale = 1 / max((ins_num - 1)^2, 1);'));
verifyTrue(testCase, contains(update_wc, 'threshold = para.gamma / denominator;'));
verifyTrue(testCase, contains(update_wd, 'para.beta * para.mu(i) * para.mu(j)'));
verifyTrue(testCase, contains(update_wd, '2 * para.eta * W_D{i}'));
end

function testTuningUsesProvidedSplitAndValidationOnly(testCase)
% Hyperparameters and threshold must be selected before the held-out test
% partition is evaluated.  A single-point grid keeps this focused test fast.
rng(7, 'twister');
n = 20;
target = double([mod(1:n, 2) == 0; mod(1:n, 3) == 0]);
data = {[target; 0.01 * randn(1, n)], ...
        [fliplr(target); 0.01 * randn(1, n)]};
split.train = 1:14;
split.validation = 15:16;
split.test = 17:20;
options.alpha_values = 1e-3;
options.gamma_values = 1e-2;
options.max_iter = 1;

result = tune_cisf_parameters(data, target, split, options);

verifyEqual(testCase, result.split, split);
verifyEqual(testCase, result.best.alpha, options.alpha_values);
verifyEqual(testCase, result.best.gamma, options.gamma_values);
verifyEqual(testCase, result.best.beta, 0.1);
verifyEqual(testCase, result.best.eta, 100);
verifyEqual(testCase, result.best.tau, 1.1);
verifyEqual(testCase, result.best.lambda0, max(1, 12 * options.alpha_values));
verifyEqual(testCase, result.best.ablation, 'full');
verifySize(testCase, result.validation.average_precision, [1, 1]);
verifyTrue(testCase, isfinite(result.validation.threshold));
verifyTrue(testCase, isfinite(result.test.metrics.HammingLoss));
verifyEqual(testCase, result.test.indices, split.test);
end

function delete_if_present(path)
if isfile(path)
    delete(path);
end
end

function p = small_test_parameters()
p = default_cisf_parameters(2);
p.max_iter = 2;
p.tol = 0;
p.eta = 1e-2;
p.beta = 1e-2;
p.gamma = 1e-3;
p.alpha = 1e-3;
end
