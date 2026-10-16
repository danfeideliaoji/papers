function [W_D] = Update_WD(X, Y, W_C, W_D, S, para)
% **********update WD*******************
% X          [cell]: 1 * v_num
% X^v      [matrix]: d_v * ins_num
% W_C        [cell]: 1 * v_num
% W^v      [matrix]: lab_num * d_v
% Y        [matrix]: lab_num * ins_num
% S        [matrix]: ins_num * ins_num
% **************************************

v_num = size(X, 2);

for i = 1 : v_num
    temp = zeros(size(Y,1), size(X{1,i},1));
    for j = 1 : v_num
        if i == j
            continue;
        end
        temp = temp + para.eta*para.mu(i)*para.mu(j)*W_D{1,j}*X{1,j}*X{1,i}';
    end
    for iter = 1 : 30
    W_D{1,i}=W_D{1,i}.*((2*para.mu(i)*S'*Y*X{1,i}'-2*para.mu(i)*S'*S*W_C{1,i}*X{1,i}*X{1,i}'-temp)./(2*para.beta*W_D{1,i}+2*para.mu(i)*S'*S*W_D{1,i}*X{1,i}*X{1,i}'));
    end
end

end


