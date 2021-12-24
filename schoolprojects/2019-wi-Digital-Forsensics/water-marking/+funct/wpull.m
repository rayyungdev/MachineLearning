function [water]= wpull(watermarkedimg,key)
watermarkedimg=double(watermarkedimg);
rng(key);
lup=rand(1,256) > .5; %lookup table of 1x256 purposes 
[row, col]=size(watermarkedimg);
water=zeros(row, col);
for i=1:row
    for k= 1:col
        wmarkpix=watermarkedimg(i,k)+1; %pixel value from 0-255
        water(i,k)=lup(wmarkpix);
    end
end
water=logical(uint8(water));
end

