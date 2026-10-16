function [C] = Update_C(X, W_C, S, M, para)
% **********update C*******************
% X          [cell]: 1 * v_num
% X^v      [matrix]: d_v * ins_num
% W_C        [cell]: 1 * v_num
% W^v      [matrix]: lab_num * d_v
% M          [cell]: 1 * v_num
% M^v      [matrix]: lab_num * ins_num
% S        [matrix]: ins_num * ins_num
% **************************************

v_num = size(X, 2);
ins_num = size(X{1,1},2);
H = ones(ins_num,ins_num)*(1/ins_num)*(-1) + eye(ins_num);
for i = 1 : v_num
    C{1,i} = S*W_C{1,i}*X{1,i};
end

for i = 1 : v_num
    K = zeros(ins_num, ins_num);
    for j = 1 : v_num
        if (j==i) 
            continue;
        end
        K =  K + H*C{1,j}'*C{1,j}*H; 
    end
    C{1,i} = (M{1,i}-para.lambda(i)*S*W_C{1,i}*X{1,i})*...
        (para.gamma*para.mu(i)*para.mu(j)*(K'+K)-para.lambda(i)*eye(ins_num))^-1;
end

end

