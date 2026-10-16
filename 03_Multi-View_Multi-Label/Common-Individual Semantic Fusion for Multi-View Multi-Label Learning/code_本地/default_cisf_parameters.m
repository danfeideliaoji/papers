function para = default_cisf_parameters(v_num)
%DEFAULT_CISF_PARAMETERS Construct the explicit default CISF configuration.
%   PARA = DEFAULT_CISF_PARAMETERS(V_NUM) returns all settings consumed by
%   the CISF training and prediction functions for V_NUM data views.

    validateattributes(v_num, {'numeric'}, {'scalar', 'integer', 'positive'}, ...
        mfilename, 'v_num');

    % These names follow the notation used in the CISF paper.
    para.alpha = 1e-2;
    para.beta = 1e-1;
    para.gamma = 1e-1;
    para.eta = 100;

    para.mu = ones(1, v_num) / v_num;
    para.lambda = ones(1, v_num);
    para.lambda_max = 1e6;
    para.tau = 1.1;
    para.thr = 0.5;
    para.max_iter = 50;
    para.tol = 1e-3;
    para.eps = 1e-10;
    % Paper ablation modes: full CISF, CISFnC, CISFnI and CISFnL.
    para.ablation = 'full';
end
