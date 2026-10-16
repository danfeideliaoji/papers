% This is the demo for TMvML
% there are three trade-off parameters \alpha, \beta, \gamma
% we set half of the original dimensionality as the size of output
% representation ----- dim_ratio
data='emotions';


para.dim_ratio=0.5;


max_val=0;
%Training
 for i = -1:1:5
        for jj = -5:1:0
            for j = -5:1:0
                para.alpha=10^(i);
                para.beta=10^(jj);
                para.gama=10^(j);
                result=our_main(data,para);
                if result(1,1) > max_val
                        max_val = result(1,1);
                        record_result = result(:,:);
                end
            end
        end
 end

tags{1,1} = 'ours';
PrintResultsAll(record_result,tags);

