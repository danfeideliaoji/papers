function Label = label_trans(Ytrn)
[n,c] = size(Ytrn);
c2 = 1/2*c*(c-1);
Label = zeros(n,c2);
k=0;
for i=1:c-1
    for j=i+1:c
        k = k+1;
        Label(:,k) = (Ytrn(:,i) - Ytrn(:,j)) / 2;
    end
end

Trans = zeros(c,c2);
j = 0;
for p=1:c-1
    for q=p+1:c
        j=j+1;
        Trans(p,j)=1;
        Trans(q,j)=-1;
    end
end
Label2 = Ytrn*Trans / 2;
end