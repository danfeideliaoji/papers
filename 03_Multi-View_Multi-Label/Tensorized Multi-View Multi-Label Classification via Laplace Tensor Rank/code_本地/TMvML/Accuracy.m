function [Acc] = Accuracy(Z,Y)
%Compute the Accuracy of the predicted annotations.
%Z is the predicted label set and Y is the true labe set. Both of which are in the 1, 0 form.
%See 5.2 of Mining Partially Annotated Images (KDD2011) for Accuracy definition.
%   written by Guoxian Yu (guoxian85@gmail.com), School of Computer Science and Engineering,
%   South China University of Technology.
%   version 1.0 date:2012-02-24
%计算预测注释的精度。

%Z是预测标签集，Y是真实标签集。两者都是1，0形式。

if min(min(Z))<0
    Z= (Z+abs(Z))/2;
end

if min(min(Y))<0
    Y= (Y+abs(Y))/2;
end

div=sum(sum(Z));
if(div==0)
    div=1;
end
Acc=sum(sum((Z.*Y)))/div;


