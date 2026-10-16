function [U,D,V] = gene_data(n,c,m)
if nargin == 0
    n = 300;
    c = 10;
    m = 200;
end
L = randn(n,c);
R = randn(m,c);
D = L*R';
L = L + 0.1*randn(n,c);
R = R + 0.1*randn(m,c);
[U, ~] = qr(L, 0);
[V, ~] = qr(R, 0);
D = D + 0.1*randn(n,m);
end