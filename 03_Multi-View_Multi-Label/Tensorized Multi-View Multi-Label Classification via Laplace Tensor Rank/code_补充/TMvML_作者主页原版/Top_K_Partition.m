function [newZ]=Top_K_Partition(Z,topK)
%partite the largest k elements as relevant and the left as irrelavant
%input  Z(Ndata x Nfun): predicted probability 
%       topK: the top K element relevant labeled as 1
%output newZ(Ndata x Nfun): partitioned relevant and irrelevant labels (1) for relevant and (-1) for irrelevant
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2012-01-08';
%将最大的k元素划分为相关元素，将左侧元素划分为非相关元素

%输入Z（Ndata x Nfun）：预测概率

%topK：标记为1的顶部K元素

%输出newZ（Ndata x Nfun）：分区的相关和不相关标签（1）表示相关，（-1）表示不相关
Z=Z';
[Ndata,Nfun]=size(Z);

newZ=full(-ones(Ndata,Nfun));
for ii=1:Ndata
    Zii=full(Z(ii,:));
    [sorted, index] = sort(Zii,'descend');
    newZ(ii,index(1:topK))=1;
end

newZ=newZ';