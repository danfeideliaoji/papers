function [posObv] = genObv2(train_target, onv_rho)
     [len_a,len_b] = size(train_target);
     alllen = len_a*len_b;
     idx = randsample(1:alllen,round(alllen*(1-onv_rho)));
     posObv = ones(len_a,len_b);
     for i=1:length(idx)
         idx_a=floor(idx(i)/len_b)+1;
         idx_b=mod(idx(i),len_b);
         if(idx_b==0)
             idx_a=idx_a-1;
             idx_b=len_b;
         end
         posObv(idx_a,idx_b)=0;
     end
end
