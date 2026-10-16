function W_C = Update_WC(X, Y, C, M, W_D, S, para)
%UPDATE_WC Update common mappings with the nuclear-norm proximal step.

v_num = numel(X);
W_C = cell(1, v_num);
for i = 1:v_num
    denominator = 2 * para.mu(i) + para.lambda(i);
    denominator = max(denominator, para.eps);
    target = (2 * para.mu(i) * (Y - S * W_D{i} * X{i}) ...
        + para.lambda(i) * C{i} + M{i}) / denominator;

    [U, singular_values, V] = svd(target, 'econ');
    singular_values = diag(singular_values);
    threshold = para.gamma / denominator;
    keep = singular_values > threshold;
    if any(keep)
        low_rank_target = U(:, keep) * diag(singular_values(keep) - threshold) * V(:, keep)';
    else
        low_rank_target = zeros(size(target));
    end

    % S and X can be rectangular or rank-deficient on a split; PINV is the
    % least-squares reconstruction used by the published update.
    W_C{i} = pinv(S) * low_rank_target * pinv(X{i});
end
end
