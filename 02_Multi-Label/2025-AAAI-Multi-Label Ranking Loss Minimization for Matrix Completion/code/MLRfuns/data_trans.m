function D2 = data_trans(D1)
gap = (max(max(D1))+min(min(D1)))/2;
D1(D1==0) = gap;
D2 = D1-gap;
end