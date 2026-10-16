function [W_C,W_D,S,para] = train_L(X, Y, para)
% **************Train*******************
% X          [cell]: 1 * v_num
% X^v      [matrix]: d_v * ins_num
% W_C, W_D   [cell]: 1 * v_num
% W^v      [matrix]: lab_num * d_v
% C, M       [cell]: 1 * v_num
% C^v, M^v [matrix]: lab_num * ins_num
% Y        [matrix]: lab_num * ins_num
% S        [matrix]: ins_num * ins_num
% **************************************

v_num = size(X,2);
[lab_num, ins_num] = size(Y);
iter = 0;
%*********initialize********************
for i = 1 : v_num
    C{1,i} = Y;
    M{1,i} = Y;
    W_D{1,i} = Y*pinv(X{1,i});
    W_C{1,i} = Y*pinv(X{1,i});
end
S = eye(lab_num);
%***************************************
last_err = 0;

while (true)
    
    %---------Update W_C----------------------
    W_C = Update_WC(X, Y, C, M, W_D, S, para);
    
    %---------Update W_D----------------------
    W_D = Update_WD(X, Y, W_C, W_D, S, para);

    %---------Update C----------------------
    C = Update_C(X, W_C, S, M, para);

    %---------Update S----------------------
    S = Update_S(X, Y, W_C, W_D, C, M, para);

    %---------Update M, lambda----------------------
    for i = 1 : v_num
        M{1,i} = M{1,i}+para.lambda(i)*(C{1,i}-S*W_C{1,i}*X{1,i});
        para.lambda(i) = min(para.lambda_max, para.tau*para.lambda(i));
        para.mu(i) = 1./(2*norm(Y-(W_C{1,i}+S*W_D{1,i})*X{1,i},"fro"));
    end
    %---------Check Stop--------------------
    iter = iter+1;
    curr_err = norm(Y-(W_C{1,i}+S*W_D{1,i})*X{1,i},"fro");
    fprintf("This is %d-th iteration, obj_err = %f \n", iter, curr_err);
    if ((abs(curr_err-last_err)/curr_err<1e-3) || (iter > 50))
        break;
    end
    last_err = curr_err;

    
end

end

