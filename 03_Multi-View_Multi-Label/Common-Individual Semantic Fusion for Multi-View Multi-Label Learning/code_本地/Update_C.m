function C = Update_C(X, W_C, S, M, para)
%UPDATE_C Update the auxiliary variables C (called A in the CISF paper).

v_num = numel(X);
ins_num = size(X{1}, 2);
H = eye(ins_num) - ones(ins_num) / ins_num;
C = cell(1, v_num);

% Use the current view reconstructions when building cross-view HSIC terms.
for i = 1:v_num
    C{i} = S * W_C{i} * X{i};
end

hsic_scale = 1 / max((ins_num - 1)^2, 1);
for i = 1:v_num
    K = zeros(ins_num, ins_num);
    for j = 1:v_num
        if j ~= i
            K = K + para.mu(j) * H * C{j}' * C{j} * H;
        end
    end

    system_matrix = para.alpha * para.mu(i) * hsic_scale * (K + K') ...
        - para.lambda(i) * eye(ins_num);
    right_hand_side = M{i} - para.lambda(i) * S * W_C{i} * X{i};

    % Solve the right-sided system by transposition instead of forming an
    % explicit inverse.  The ridge is used only for a numerically singular
    % system.
    if rcond(system_matrix) < para.eps
        system_matrix = system_matrix - para.eps * eye(ins_num);
    end
    C{i} = (system_matrix' \ right_hand_side')';
end
end
