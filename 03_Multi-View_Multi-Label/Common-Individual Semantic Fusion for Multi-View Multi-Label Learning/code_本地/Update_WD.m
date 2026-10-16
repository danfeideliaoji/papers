function W_D = Update_WD(X, Y, W_C, W_D, S, para)
%UPDATE_WD Update individual mappings by safeguarded multiplicative steps.
%   BETA weights the cross-view individual-mapping term and ETA weights the
%   individual-mapping Frobenius term, following the paper's notation.

v_num = numel(X);
safe_eps = max(para.eps, eps);
for i = 1:v_num
    X_i = X{i};
    cross_view = zeros(size(Y, 1), size(X_i, 1));
    for j = 1:v_num
        if j ~= i
            cross_view = cross_view + para.beta * para.mu(i) * para.mu(j) ...
                * W_D{j} * X{j} * X_i';
        end
    end

    W_D{i} = max(real(W_D{i}), 0);
    for iteration = 1:30 %#ok<NASGU>
        numerator = 2 * para.mu(i) * S' * Y * X_i' ...
            - 2 * para.mu(i) * S' * S * W_C{i} * X_i * X_i' ...
            - cross_view;
        denominator = 2 * para.eta * W_D{i} ...
            + 2 * para.mu(i) * S' * S * W_D{i} * X_i * X_i';

        ratio = max(real(numerator), safe_eps) ./ max(real(denominator), safe_eps);
        W_D{i} = max(real(W_D{i} .* ratio), 0);
    end
end
end
