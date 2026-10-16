function HP = updateHPabsentClusteringOrthHp(WP,Hstar,S,HP00)

num = size(Hstar,1);
k = size(Hstar,2);
numker = size(WP,3);
HP = zeros(num,k,numker);
for p = 1:numker
    mis_indx = S{p}.indx';
    obs_indx = setdiff(1:num,mis_indx);
    %%%%
    Vp = Hstar(mis_indx,:)*WP(:,:,p)';
    [Up,Sp,Vp] = svd(Vp,'econ');
    if length(mis_indx) > 0
        HP(mis_indx,:,p) = Up*Vp';
    end
    %%%%
    HP(obs_indx,:,p) = HP00(obs_indx,:,p);
end