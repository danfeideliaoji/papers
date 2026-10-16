% ***************Run*******************
% data       [cell]: 1 * v_num
% data^v   [matrix]: d_v * ins_num
% target   [matrix]: lab_num * ins_num
% W_C, W_D   [cell]: 1 * v_num
% W^v      [matrix]: lab_num * d_v
% C, M       [cell]: 1 * v_num
% C^v, M^v [matrix]: lab_num * ins_num
% Y        [matrix]: lab_num * ins_num
% S        [matrix]: lab_num * lab_num
% **************************************
addpath(genpath('metrics'));
addpath(genpath('dataset'));
load('emotions.mat');
v_num = size(data,2);


data = data';
for i = 1 : v_num
    data{i,1} = data{i,1}';
    data{i} = normalization(data{i}, 'l2', 1);
end

data = data';
for i = 1 : v_num
    data{1,i} = data{1,i}';
end



[row,col]=size(data{1,1});
encode=crossvalind('Kfold',data{1,1}(row,1:col),5);
HL = []; RL = []; OE = []; Cov = []; AP = [];
for i = 1 : 5
    % data processing...
    test = (encode == i);
    for v = 1 : v_num
        %data{1,v} = zscore(data{1,v});
        train_data{1,v} = data{1,v}(:,find(test==0));
        test_data{1,v} = data{1,v}(:,find(test==1));
    end
    train_target = target(:,find(test==0));
    test_target = target(:,find(test==1));
    
    [W_C, W_D, S, para] = train_L(train_data, train_target, para);
    [P] = test_L(test_data, W_C, W_D, S, para);

PP = P;
PP(P>para.thr) = 1;
PP(P<=para.thr) = 0;

HL=[HL, Hamming_loss(PP,test_target)];
RL=[RL, Ranking_loss(P,test_target)];
OE=[OE, One_error(P,test_target)];
Cov=[Cov, coverage(P,test_target)];
AP=[AP, Average_precision(P,test_target)];
end

