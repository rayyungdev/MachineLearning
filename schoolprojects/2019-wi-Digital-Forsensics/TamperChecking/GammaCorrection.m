function [NewPixel] = GammaCorrection(Pixels,gamma,on)
[pRows, pCols] = size(Pixels);
x=zeros(pRows, pCols);
for i = 1:pCols
    for j = 1: pRows;
        pout_pixel=double(Pixels(j, i));
        new_pout=255*power((pout_pixel/255),gamma);
        x(j,i)=new_pout;
    end
end
NewPixel=uint8(x);

if on==1
    subplot(1,2,1);
    imshow(Pixels);
    title('Original')
    subplot(1,2,2);
    imshow(NewPixel);
    title(['Gamma: ', num2str(gamma)]);
end
end

