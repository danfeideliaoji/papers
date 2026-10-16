function [value] = evaluate_PR(predict,target,flg)
    if(size(predict) ~= size(target))
        disp('error');
    end
    [dimData,numLabel] = size(predict);
    seq = getseq(predict);
    rank = getrank(seq);

    rankingloss=0;
    avg_precision=0;
    for i=1:dimData
        dim_ti = getdimm(target(i,:));
        
        if(dim_ti~=0 && dim_ti~=numLabel)
            cnt_rank = 0;
            for j=1:numLabel
                if(target(i,j)==0)
                    continue
                end
                for k=1:numLabel
                    if(target(i,k)==1)
                        continue
                    end
                    if(predict(i,j) <= predict(i,k))
                        cnt_rank = cnt_rank+1;
                    end
                end
            end
            rankingloss = rankingloss + cnt_rank/(dim_ti*(numLabel-dim_ti));
        end

        if(dim_ti == numLabel || dim_ti == 0)
            avg_precision = avg_precision+1;
        else
            cnt_pre = 0;
            for j=1:numLabel
                if(target(i,j)==0)
                    continue
                end
                tmp = 0;
                for k=1:numLabel
                    if(target(i,k)==0)
                        continue
                    end
                    if(rank(i,j) >= rank(i,k))
                        tmp = tmp+1;
                    end
                end
                cnt_pre = cnt_pre + tmp/rank(i,j);
            end
            avg_precision = avg_precision + cnt_pre/dim_ti;
        end
    end
    rankingloss = rankingloss/dimData;
    avg_precision = avg_precision/dimData;
    if(flg=="rl")
        value = rankingloss;
    else
        value = avg_precision;
    end
end

function [dimm] = getdimm (xi)
    num = length(xi);
    dimm = 0;
    for i = 1:num
        if(xi(i)>=0.99)
            dimm = dimm+1;
        end
    end
end

%{0.45, 0.66, 0.73, 0.29, 0.80} -> {4, 2, 1, 0, 3}
function [seq] = getseq(y)
    [dim,num] = size(y);
    seq = zeros(dim,num);
    for i = 1:dim
        seq(i,:) = arraysort(y(i,:));
    end
end

%{0.45, 0.66, 0.73, 0.29, 0.80} ->> {4, 3, 2, 5, 1}
function [rank] = getrank(seq)
	[dim,num] = size(seq);
    rank = zeros(dim,num);
    for i=1:dim
        for j=1:num
            rank(i,seq(i,j)) = j;
        end
    end
end

function [index] = arraysort(org_arr)
    num = length(org_arr);
    new_arr = 1:1:num;
    for i=1:num
        new_arr(i) = org_arr(i);
    end
    index = 1:1:num;
    for i=1:num
        for j=1:num-i
            if new_arr(j) < new_arr(j+1)
                tmp = new_arr(j);
                new_arr(j) = new_arr(j+1);
                new_arr(j+1) = tmp;
                
                tmpindex = index(j);
                index(j) = index(j+1);
                index(j+1) = tmpindex;
            end
        end
    end
end
