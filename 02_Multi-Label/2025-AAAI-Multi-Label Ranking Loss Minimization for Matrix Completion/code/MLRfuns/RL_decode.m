function Y_hat = RL_decode(L_hat)
[m,c2] = size(L_hat);
c = 1/2+sqrt(2*c2+1/4);
Y_hat = zeros(m,c);
for i=1:c
    for j=1:c-1
        if(j<i)
            Y_hat(:,i) = Y_hat(:,i) - L_hat(:,sn(j-1,c)+i-j);
        else
            Y_hat(:,i) = Y_hat(:,i) + L_hat(:,sn(i-1,c)+j-i+1);
        end
    end
    Y_hat(:,i) = Y_hat(:,i)/(c-1);
end
end

function out = sn(n,c)
out = n*(2*c-n-1)/2;
end
