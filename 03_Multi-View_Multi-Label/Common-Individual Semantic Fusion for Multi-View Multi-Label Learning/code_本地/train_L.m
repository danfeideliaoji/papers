function [W_C, W_D, S, para] = train_L(X, Y, para)
%TRAIN_L Train CISF with paper-consistent reconstruction residuals.

v_num = numel(X);
[lab_num, ~] = size(Y);
para.ablation = cisf_ablation_mode(para);
remove_common = strcmp(para.ablation, 'no_common');
remove_individual = strcmp(para.ablation, 'no_individual');
fix_label_correlation = strcmp(para.ablation, 'no_label_correlation');
C = cell(1, v_num);
M = cell(1, v_num);
W_C = cell(1, v_num);
W_D = cell(1, v_num);

for i = 1:v_num
    C{i} = Y;
    M{i} = Y;
    initial_mapping = Y * pinv(X{i});
    if remove_common
        W_C{i} = zeros(size(initial_mapping));
        C{i} = zeros(size(Y));
        M{i} = zeros(size(Y));
    else
        W_C{i} = initial_mapping;
    end
    % The multiplicative W_D update assumes nonnegative factors.
    if remove_individual
        W_D{i} = zeros(size(initial_mapping));
    else
        W_D{i} = max(real(initial_mapping), 0);
    end
end
S = eye(lab_num);
last_err = inf;

for iter = 1:para.max_iter
    if remove_common
        W_C = zero_mappings(X, lab_num);
        C = zero_semantics(Y, v_num);
        M = zero_semantics(Y, v_num);
    else
        W_C = Update_WC(X, Y, C, M, W_D, S, para);
    end
    if remove_individual
        W_D = zero_mappings(X, lab_num);
    else
        W_D = Update_WD(X, Y, W_C, W_D, S, para);
    end
    if ~remove_common
        C = Update_C(X, W_C, S, M, para);
    end
    if fix_label_correlation
        S = eye(lab_num);
    else
        S = Update_S(X, Y, W_C, W_D, C, M, para);
    end

    residual_norms = zeros(1, v_num);
    for i = 1:v_num
        residual = Y - S * (W_C{i} + W_D{i}) * X{i};
        residual_norms(i) = norm(residual, 'fro');
    end
    % Measure with the weights that generated this iteration.  Updating MU
    % first would make every nonzero term identically one half.
    curr_err = sum(para.mu .* residual_norms);

    for i = 1:v_num
        residual = Y - S * (W_C{i} + W_D{i}) * X{i};
        if ~remove_common
            M{i} = M{i} + para.lambda(i) * (C{i} - S * W_C{i} * X{i});
            para.lambda(i) = min(para.lambda_max, para.tau * para.lambda(i));
        end
        para.mu(i) = 1 / (2 * max(norm(residual, 'fro'), para.eps));
    end
    para.mu = para.mu / max(sum(para.mu), para.eps);

    fprintf('This is %d-th iteration, weighted_residual = %f\n', iter, curr_err);
    if iter > 1 && abs(curr_err - last_err) / max(abs(curr_err), para.eps) < para.tol
        break;
    end
    last_err = curr_err;
end
end

function mappings = zero_mappings(X, lab_num)
v_num = numel(X);
mappings = cell(1, v_num);
for view = 1:v_num
    mappings{view} = zeros(lab_num, size(X{view}, 1));
end
end

function semantics = zero_semantics(Y, v_num)
semantics = cell(1, v_num);
for view = 1:v_num
    semantics{view} = zeros(size(Y));
end
end
