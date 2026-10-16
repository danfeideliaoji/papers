function S = Update_S(X, Y, W_C, W_D, C, M, para)
%UPDATE_S Update the shared label-correlation matrix using every view.
%   This implements the accumulated normal equation in CISF equation (11).

v_num = numel(X);
[lab_num, ~] = size(Y);
S_A = zeros(lab_num, lab_num);
S_B = zeros(lab_num, lab_num);

for i = 1:v_num
    X_i = X{i};
    W_C_i = W_C{i};
    T_i = W_C_i + W_D{i};
    S_A = S_A + 2 * para.mu(i) * Y * X_i' * T_i' ...
        + para.lambda(i) * C{i} * X_i' * W_C_i' ...
        + M{i} * X_i' * W_C_i';
    S_B = S_B + 2 * para.mu(i) * T_i * X_i * X_i' * T_i' ...
        + para.lambda(i) * W_C_i * X_i * X_i' * W_C_i';
end

% A small diagonal ridge prevents a singular normal equation without
% changing the per-view accumulation above.
ridge = max(para.eps, eps(class(S_B)));
S = S_A / (S_B + ridge * eye(lab_num));
S = max(real(S), 0);

% Normalize columns before imposing the model's unit-diagonal constraint.
% Setting the diagonal afterwards deliberately means final columns are not
% claimed to have unit l1 norm.
column_sums = max(sum(S, 1), ridge);
S = S ./ column_sums;
S(1:lab_num+1:end) = 1;
end
