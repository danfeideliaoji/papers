function [M_recover, M, N, iter] = MLR_global(R_Omega, U, V, lambda1, lambda2, maxiter, tol, LL)
if(max(max(R_Omega))+min(min(R_Omega))~=0)
    disp("Error: Lack of normalization. ----PRLtrain.m");
end
scaling = max(max(R_Omega))*2;
R_Omega = R_Omega*LL / scaling;
B = (V'*LL)';
%% Step1: Initialization
[n, m2] = size(R_Omega);
r_a = size(U, 2);
r_b = size(V, 2);
M = zeros(n, m2);
N = zeros(r_a, r_b);

%% Step2: Iteration
i = 1;
stop = 1;
matr = M + U * N * B';

L_p = sqrt(norm(U'*U, 'fro')*norm(B'*B, 'fro'));
while(stop > tol)
    matr_1 = matr;
    % update M
    E = U * N * B' + R_Omega;
    M_1 = fsvt(E, lambda1);
    
    % update N
    F = N - 1/L_p * U' * (U * N * B' - M_1) * B;
    N_1 = fL1(F, lambda2);
    
    matr = M_1 + U * N_1 * B';
    stop = norm(matr - matr_1, 'fro') / norm(matr_1, 'fro');
    M = M_1;
    N = N_1;
    i=i+1;
    
    if i < maxiter
        iter = i - 1;
    else
        iter = maxiter;
        break
    end
end

%% Step3: The completed matrix
M_recover = U * N * B' + M;

M_recover = M_recover*pinv(LL) * scaling;
end
