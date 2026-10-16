function dd = correct(target,Y_hat,obrT)
tmp = max(sum(obrT, 2),1); % if no items are observed in some lines, at least 1
dd = (sum((target-Y_hat).*obrT, 2)./tmp) * ones(1,size(target,2));
end