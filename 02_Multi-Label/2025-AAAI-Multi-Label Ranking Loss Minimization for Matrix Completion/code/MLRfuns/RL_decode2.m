function Y_hat = RL_decode2(L_hat)
[n,c2] = size(L_hat);
c = 1/2+sqrt(2*c2+1/4);
Trans = zeros(c,c2);
j = 0;
for p=1:c-1
    for q=p+1:c
        j=j+1;
        Trans(p,j)=1;
        Trans(q,j)=-1;
    end
end
Y_hat = L_hat*pinv(Trans);
end