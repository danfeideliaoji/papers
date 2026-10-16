function [M] = project_gradient_descent(M, grad, eta)
    % 对M进行奇异值分解
    [U, s, V] = svd(M);
    % 计算s的逆矩阵
    s_inv = zeros(size(s));
    s_inv(1:length(s)) = 1.0 ./ s(1:length(s));
    % 计算投影矩阵P
    P = U * diag(diag(s_inv)) * U';
    % 将梯度投影到正交矩阵空间中
    grad_proj = P * grad;
    % 使用投影后的梯度更新M
    M = M - eta * grad_proj;
    % 对更新后的M进行正交化处理
    [Q, ~] = qr(M);
    M = Q;
end



