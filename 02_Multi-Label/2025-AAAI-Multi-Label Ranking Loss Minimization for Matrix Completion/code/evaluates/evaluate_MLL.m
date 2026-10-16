function [value] = evaluate_MLL(predict,target,threshold)
    if(size(predict) ~= size(target))
        disp('error');
    end
    [dimData,numLabel] = size(predict);
    % Step 1: Calculation for ranking-based measures
    seq = getseq(predict);
    rank = getrank(seq);

    oneerror=0;
    coverage=0;
    rankingloss=0;
    avg_precision=0;
    for i=1:dimData
        dim_ti = getdimm(target(i,:));
        
        if(round(target(i, seq(i,1))) ~= 1)
            oneerror = oneerror+1;	%error, when the most confident one is incorrectly estimated
        end
        cnt_cov = dim_ti;
        r=1;
        while(r<=numLabel && cnt_cov~=0)
            if( target(i, seq(i,r))==1 )
                cnt_cov = cnt_cov-1;
            end
            r = r+1;
        end
        coverage = coverage+r;
        
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
    oneerror = oneerror/dimData;
    coverage = coverage/dimData - 1;
    coverage = coverage/numLabel;
    rankingloss = rankingloss/dimData;
    avg_precision = avg_precision/dimData;
    
    % Step 2: Calculation for classification-based measures
    predict = rint(predict,threshold);
    
    acc=0;
    precision=0;
    recall=0;
    f1=0;
    hamming=0;
    hitrate=0;
    subsetAcc=0;
    
    for i = 1:dimData
        a=0;b=0;c=0;d=0;
        for j = 1:numLabel
            if(predict(i,j)==1 && target(i,j)==1)
                a = a+1;
            end
            if(predict(i,j)==1 && target(i,j)==0)
                c = c+1;
            end
            if(predict(i,j)==0 && target(i,j)==1)
                b = b+1;
            end
            if(predict(i,j)==0 && target(i,j)==0)
                d = d+1;
            end
        end
%         disp([a,b,c,d]);
        if(a+b+c==0)
            acc = acc+1;
        else
            acc = acc + a/(a+b+c);
        end
        if(a>0)
            hitrate = hitrate+1;
        end
        if(a+b+c == 0)
            precision = precision+1;
            recall = recall+1;
        else
            if(a+c ~= 0)
                precision = precision + a/(a+c);
            end
            if(a+b ~= 0)
                recall = recall + a/(a+b);
            end
        end
        if(2*a+b+c==0)
            f1 = f1+1;
        else
            f1 = f1 + 2*a/(2*a+b+c);
        end
        hamming = hamming + (b+c)/(a+b+c+d);
        if(b==0 && c==0)
            subsetAcc = subsetAcc+1;
        end
    end
    acc = acc/dimData;
    precision = precision/dimData;
    recall = recall/dimData;
    f1 = f1/dimData;
    hitrate = hitrate/dimData;
    subsetAcc = subsetAcc/dimData;
    hamming = hamming/dimData;
    
    value = [acc, precision, recall, f1,  hitrate, subsetAcc, hamming, oneerror, coverage, rankingloss, avg_precision];
end
            
function [y] = rint(x,threshold)
    [dim,num] = size(x);
    y = zeros(dim,num);
    for i = 1:dim
        for j = 1:num
            if(x(i,j)>=threshold)
                y(i,j)=1;
            else
                y(i,j)=0;
            end
        end
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