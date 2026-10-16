function [M_recover] = MLR_trans(R, lambda, LL)
R = R*LL;
M_1 = fsvt(R, lambda);
M_recover = M_1*pinv(LL);
end
