function [P] = test_L(X, W_C, W_D, S, para)
% ***************Test*******************
% X          [cell]: 1 * v_num
% X^v      [matrix]: d_v * ins_num
% W_C        [cell]: 1 * v_num
% W^v      [matrix]: lab_num * d_v
% M          [cell]: 1 * v_num
% M^v      [matrix]: lab_num * ins_num
% S        [matrix]: ins_num * ins_num
% **************************************
v_num = size(X,2);
P = zeros(size(W_C{1,1},1),size(X{1,1},2));
for i = 1 : v_num
    P = P + para.mu(i)*S*(W_C{1,i}+W_D{1,i})*X{1,i};
end

end