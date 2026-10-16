function [posGiven, posSample] = misObv2(DATA, sr)
% sr: sampling ratio

posGiven = double(DATA ~= 0);
tmp = DATA';
pos_given = find(tmp ~= 0);
num_given = sum(sum(posGiven));
num_sample = floor(num_given * sr);
a = randperm(num_given);
a = a(1:num_sample);
pos_sample = pos_given(a);
[dim1,dim2] = size(DATA);
posSample = zeros(dim1,dim2);
for zzz = 1:length(pos_sample)
    idx = pos_sample(zzz);
    idx_a=floor(idx/dim2)+1;
    idx_b=mod(idx,dim2);
    if(idx_b==0)
        idx_a=idx_a-1;
        idx_b=dim2;
    end
    posSample(idx_a,idx_b)=1;
end
end