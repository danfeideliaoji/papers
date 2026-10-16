function y = rank_fun_derivative1(x,delta)
%y  = exp(-x./(delta))./(delta);
y  = exp(-x./(delta+eps))./(delta+eps)*exp(delta);
%y = delta./(delta+x).^2;end*exp(delta^2)
