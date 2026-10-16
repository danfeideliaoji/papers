function Trans = getTransMat(c, rat)
c2 = 1/2*c*(c-1);
Trans = zeros(c,c2);
j = 0;
for p=1:c-1
    for q=p+1:c
        j=j+1;
        Trans(p,j)=1;
        Trans(q,j)=-1;
    end
end
if nargin==2
    if rat==1
        return;
    end
    a = randperm(c2-c+1);
    a = a(1:int32(c2*rat));
    a = [1:c-1,a];
    Trans = Trans(:,a);
end
end


