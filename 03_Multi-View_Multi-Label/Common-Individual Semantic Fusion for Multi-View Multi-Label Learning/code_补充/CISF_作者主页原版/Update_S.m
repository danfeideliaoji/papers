function [S] = Update_S(X, Y, W_C, W_D, C, M, para)
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
[lab_num, ins_num] = size(Y);
S = zeros(lab_num, lab_num);
for i = 1 : v_num
    T0 = (W_C{1,i}+W_D{1,i});
    S_A = (2*para.mu(i)*Y*X{1,i}'*T0'+para.lambda(i)*C{1,i}*X{1,i}'*W_C{1,i}'+M{1,i}*X{1,i}'*W_C{1,i}');
    S_B = (2*para.mu(i)*T0*X{1,i}*X{1,i}'*T0'+para.lambda(i)*W_C{1,i}*X{1,i}*X{1,i}'*W_C{1,i}');
    S_B_len = size(S_B);
    S=S_A/(S_B+eye(S_B_len(1)));
    S(S<0)=0;
    s_norms=sum(abs(S),1);
    norms = max(s_norms,1e-30);
    S=abs(S)./repmat(norms,size(S,1),1);
    S=S-diag(diag(S))+diag(ones(size(S,1),1));
end
% S = mapminmax(S,0,1);
% S = (S+S')/2;
end

