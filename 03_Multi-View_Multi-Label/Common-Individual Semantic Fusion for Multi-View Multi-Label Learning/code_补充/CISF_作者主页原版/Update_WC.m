function [W_C] = Update_WC(X,Y,C,M,W_D,S,para)
% **********update WC*******************
% X          [cell]: 1 * v_num
% X^v      [matrix]: d_v * ins_num
% W_C, W_D   [cell]: 1 * v_num
% W^v      [matrix]: lab_num * d_v
% C, M       [cell]: 1 * v_num
% C^v, M^v [matrix]: lab_num * ins_num
% Y        [matrix]: lab_num * ins_num
% S        [matrix]: ins_num * ins_num
% **************************************

v_num = size(X, 2);

for i = 1 : v_num
    temp = (1/(2*para.mu(i)+para.lambda(i)))...
        *(2*para.mu(i)*(Y-S*W_D{1,i}*X{1,i})+para.lambda(i)*C{1,i}+M{1,i});

    [U,A,V] = svd(temp,'econ');
    sp = diag(A);
    svp = length(find(sp>para.alpha/(para.lambda(i)+2*para.mu(i))));
    if svp>=1
        sp = sp(1:svp)-para.alpha/(para.lambda(i)+2*para.mu(i));
    else
        svp=1;
        sp=0;
    end
    W_C{1,i} = pinv(S)* U(:,1:svp)*diag(sp)*V(:,1:svp)' *pinv(X{1,i});
end

end

