function [bestpara, precision, record] =ParaSearch(dataset, para_range, para, crossfoldtime)
label = dataset.target;
X = dataset.data;

[N,~]=size(label);
round=crossfoldtime;
Accuracys=zeros(1,round);
AveragePrecisions=zeros(1,round);
topK=ceil(sum(sum(label))/size(label,1));

batch_num = floor(N*1/crossfoldtime);
index=randperm(N);
record = zeros(length(para_range.lambda2range) * length(para_range.lambda3range), 3 + 2 + 10);
train_index = zeros(round, batch_num);
test_index = zeros(round, N - batch_num);
for run = 1 : round
    train_index(run, :)=index(batch_num * (run-1) + 1 : batch_num * run);
    test_index(run, :) = setdiff(1:N, train_index(run, :));
end
%%  train LateFusionMVML algorithm

t = 1;
for i1 = 1 : length(para_range.lambda1range)
    for i2 = 1 : length(para_range.lambda2range)
        for i3 = 1 : length(para_range.lambda3range)
            record(t, 1) = para_range.lambda1range(i1);
            record(t, 2) = para_range.lambda2range(i2);
            record(t, 3) = para_range.lambda3range(i3);
            t = t + 1;
        end
    end
end
save('record_of co lowrank Corel5k.mat', 'record');
% load('record_of co lowrank Corel5k.mat');
% recordfile= matfile('record_of co lowrank Corel5k.mat');
% recordfile.Properties.Writable = true;

for t = 1 : size(record, 1)
    para.lambda = record(t, 1:3);
        parfor run=1 : round
            test_Y_01=label(test_index(run, :),:);
            [Y_pred_in01, ~] = BipartiteGraph_co_lowrankClas(X, label, train_index(run, :), para);
            if  abnormal == 1
               Accuracys(run) = 0;
               AveragePrecisions(run) = 0;
            else        
                Y_pred_01 = Top_K_Partition(Y_pred_in01',topK)';
                test_Y=2*test_Y_01-ones(size(test_Y_01));   
                Accuracys(run)=Accuracy(Y_pred_01,test_Y_01);
                AveragePrecisions(run) = Average_precision(Y_pred_in01',test_Y');
            end
        end
    record(t, 4 : 5) =  [sum(Accuracys) / round, sum(AveragePrecisions) / round];
    record(t, 6 : end) = [Accuracys, AveragePrecisions];
    recordfile.record(t, 4 : 5) = [sum(Accuracys) / round, sum(AveragePrecisions) / round];
    recordfile.record(t, 6 : end) = [Accuracys, AveragePrecisions];
     if ismember(0, Accuracys)
        recordfile.record(t, end) = 1;
     end
    fprintf('Running t = %d AC=%d, AP=%d\n', t, sum(Accuracys) / round, sum(AveragePrecisions) / round);
end
precisions = (record(:, 4) + record(:, 5)) / 2;
[precision, idx] = max(precisions);
bestpara = para;
bestpara.lambda = record(idx, 1 : 3);
% bestpara.k = record(idx, end-1);

end