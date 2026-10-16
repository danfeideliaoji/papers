function [Avg_Result] =our_main(dataset,para)

load(dataset);
if(min(min(target))<=-1)  %标签如果存在小于0的，就把他置为0
    target(target<0)=0;
end
Dim  =128;%最大维度
para.dim=Dim*para.dim_ratio;%计算k值
para.dim=fix(para.dim);
[mNum,~]=size(target);
round=3;
newdata=data;

Accuracys=zeros(1,round);
RankingLosses=zeros(1,round);
AveragePrecisions=zeros(1,round);
AUCs=zeros(1,round);

times=zeros(1,round);
partition_point=ceil(mNum*0.8);
m=length(newdata);

for run=1:round
    fprintf('Running %d / %d\n', run, round);
        rand1 = randperm(mNum,mNum);
for i=1:m

    v1 = newdata{i,1};
    newdata{i,1} = v1(rand1,:);
end
v2 = target;
target = v2(rand1,:);

    t0=clock;  %获取当前时间
    X=newdata;
    label=target;
    index = (1:mNum);
    train_index=index(1:partition_point);   %训练集
    test_index=index(partition_point+1:end);%测试集
    test_target=label(test_index,:);
    topK=ceil(sum(sum(label))/size(label,1));
    EstiY=label;
    %%  train ICM2L algorithm
    Outputs_all=our_Clas23(X,EstiY,train_index,para);
    
    %Outputs=Outputs_all(test_index,:);%取出测试集label
    Outputs=Outputs_all;
    times(1,run)=etime(clock,t0);%时间
    Y=test_target;    
    rocZ=Outputs;          
    index=find(sum(Y,2)==0);
 
    Y(index,:)=[];%列相加为0的去除
    rocZ(index,:)=[];
    Z=Top_K_Partition(rocZ',topK)';
    newY=2*Y-ones(size(Y));   
    Accuracys(1,run)=Accuracy(Z,Y);
    RankingLosses(1,run) = 1-Ranking_loss(rocZ',newY');
    AveragePrecisions(1,run) = Average_precision(rocZ',newY');
    [tpr,fpr] = mlr_roc(rocZ, newY);
    [AUC, ~] = mlr_auc(fpr,tpr);
    AUCs(1,run)=AUC;
    %%
    result = zeros(5,1);
    Pre_Labels = rocZ;
    Pre_Labels=mapminmax(Pre_Labels,0,1);
    Pre_Labels(Pre_Labels>0.7)=1;
    Pre_Labels(Pre_Labels<=0.7)=0;
    result(:,1)  = EvaluationAll(Pre_Labels',rocZ',Y');
    results{run}=result;
    
    
    
    
    
    
    
    


end

Avg_Result = zeros(5,2);
cvResult = zeros(5,round);
tags = cell(1,1);

for j = 1:round
    cvResult(:,j) = results{j}(:,1);
end
Avg_Result(:,1)=mean(cvResult,2);
Avg_Result(:,2)=std(cvResult,1,2);
% tags{1,1} = 'ours';
% PrintResultsAll(Avg_Result,tags);
fprintf(['training' ...
    '--------------------------------------------\n']);





end