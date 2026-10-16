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
v_num = numel(X);
mode = cisf_ablation_mode(para);
remove_common = strcmp(mode, 'no_common');
remove_individual = strcmp(mode, 'no_individual');
if strcmp(mode, 'no_label_correlation')
    S = eye(size(W_C{1}, 1));
end
P = zeros(size(W_C{1,1},1),size(X{1,1},2));
for i = 1 : v_num
    mapping = zeros(size(W_C{i}));
    if ~remove_common
        mapping = mapping + W_C{i};
    end
    if ~remove_individual
        mapping = mapping + W_D{i};
    end
    P = P + para.mu(i) * S * mapping * X{i};
end

end
