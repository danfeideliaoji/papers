function [M_recover] = MLR_noisy(R, U, V, lambda1, lambda2, lambda3, maxiter, tol, LL)
if(max(max(R))+min(min(R))~=0)
    disp("Error: Lack of normalization. ----PRLtrain.m");
end
R = R*LL;
B = (V'*LL)';
%% Step1: Initialization
[n, m2] = size(R);
r_a = size(U, 2);
r_b = size(V, 2);
M = zeros(n, m2);
N = zeros(r_a, r_b);
Z = zeros(n, m2);

%% Step2: Iteration
i = 1;
stop = 1;
matr = M +U * N * B' + Z;

while(stop > tol)
    matr_1 = matr;
    % update Z
    Z = (R-M-U * N * B')/(lambda3+1);
    
    % update M
    E = (R - U * N * B' - Z)/2;
    M = fsvt(E, lambda1);
    
    % update N
    L_p = sqrt(norm(U'*U, 'fro')*norm(B'*B, 'fro'));
    F = N - 1/L_p*(U'*(M + U * N * B' + Z - R)*B);
    N = fL1(F, lambda2);
    
    matr = M+U * N * B' + Z;
    stop = norm(matr - matr_1, 'fro') / norm(matr_1, 'fro');
    % M = M;
    % N = N;
    i=i+1;
    
    if i > maxiter
        break
    end
end

%% Step3: The completed matrix
M_recover = U * N * B' + M + Z;
M_recover = M_recover*pinv(LL);
end
