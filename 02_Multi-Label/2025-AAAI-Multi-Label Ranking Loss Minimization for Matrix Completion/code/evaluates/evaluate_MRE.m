% Mean Recovery Error
% function [MRE] = evaluate_MRE(predict,target,threshold,obvmat) % threshold for only predict, target must {0,1}
%     predict = rint(predict,threshold);
%     A = predict==target;
%     cnts = A.*(1-obvmat);
%     acc = sum(cnts(:))/sum(sum((1-obvmat)));
%     MRE = 1-acc;
% end

function [MRE] = evaluate_MRE(predict,target,threshold,posTest) % threshold for only predict, target must {0,1}
    predict = rint(predict,threshold);
    A = predict==target;
    cnts = A.*(posTest);
    acc = sum(cnts(:))/sum(sum((posTest)));
    MRE = 1-acc;
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