function [K] = tampcheck(pixels)
pixels=double(pixels);
seg=segm(pixels); %returns 8x8xNumberOfSegments
[~, colN]=size(pixels); 
[~, ~, blck]=size(seg);
ccolN=colN/8; %only the column
Z1=seg(4:5,4:5,:); %middle values
Z2=zeros(2,2,blck); %array
for k=1:blck
    if rem(k,ccolN) == 0 %shifts last column
        Z2(:,:,k)=Z2(:,:,k-1);
    elseif k > blck-ccolN %%shifts last row
        Z2(:,:,k)=Z2(:,:,k-ccolN);
    else
        Z2(1,1,k)=seg(8,8,k); %E
        Z2(2,1,k)=seg(1,8,k+ccolN);%G
        Z2(1,2,k)=seg(8,1,k+1); %F
        Z2(2,2,k)=seg(1,1,k+1+ccolN); %H
    end    
end
Z1P=zeros(1,blck);
Z2P=zeros(1,blck);
for f=1:blck
    Z1P(f)=abs(Z1(1,1,f)-Z1(1,2,f)-Z1(2,1,f)+Z1(2,2,f));
    Z2P(f)=abs(Z2(1,1,f)-Z2(1,2,f)-Z2(2,1,f)+Z2(2,2,f));
end
[hp,c1]=hist(Z1P);
[hpp,c2]=hist(Z2P);
hp=hp/blck;
hpp=hpp/blck;
K=abs(hp-hpp);
K=sum(K);
figure
subplot(1,2,1)
bar(c1,hp);
title("Z'") 
subplot(1,2,2)
bar(c2,hpp);
title("Z''")
end