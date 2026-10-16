
function normdata = normalization(data, type, addEPS)
% Data Normalization
%    Syntax
%
%       normdata = normalization(data, type, addEPS)
%
%    Input 
%       data        - n by d matrix
%       type        - 'minmax',  xij = (xij - min)/(max-min)
%                     'meanstd', xij = (xij - mean)/std
%                     'l2', norm(xi) = 1
%       addEPS      - {0,1}, data + eps
%
%    Output
%       normdata    - Normarlized Data Matrix


    
    if strcmp(type, 'minmax')
        [n,d] = size(data);
        normdata = zeros(n,d);
        for i = 1:d
            minvalue = min(data(:,i));
            maxvalue = max(data(:,i));

            denominator = max(maxvalue - minvalue, eps);
            normdata(:,i) = (data(:,i) - ones(n,1)*minvalue)./denominator;
        end
    elseif strcmp(type, 'meanstd')
        [n,d] = size(data);
        normdata = zeros(n,d);
        for i = 1:d
            meanvalue = mean(data(:,i));
            stdvalue = std(data(:,i));

            denominator = max(stdvalue, eps);
            normdata(:,i) = (data(:,i) - ones(n,1)*meanvalue)./denominator;
        end
    elseif strcmp(type, 'l2')
        if addEPS
            data = data + eps;
        end
        row_norm = max(sqrt(sum(data.^2,2)), eps);
        normdata = data./repmat(row_norm,1,size(data,2));
        normdata = [normdata,ones(size(data,1),1)];
    else
        error('CISF:UnknownNormalization', 'Unknown normalization type: %s', type);
    end


end
