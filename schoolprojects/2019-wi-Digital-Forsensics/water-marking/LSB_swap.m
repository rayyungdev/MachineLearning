% 1. Create watermarked lsb baboon with barbara
 baboon_wm = funct.LSB('..\data\baboon.tif','..\data\Barbara.bmp',1);
% 2. Create watermarke lsb pepeprs with barbara
 peppers_wm = funct.LSB('..\data\peppers.tif','..\data\Barbara.bmp',1);
 
 %Output all bit layers of watermarked baboon image 
 figure
 suptitle('Watermarked Baboon.tif Bit Layer')
 for jj = 1:8
    bp = bitget(baboon_wm, jj); 
    subplot(2,4,jj)
    imshow(bp, []);
    title(['Layer ' num2str(jj)])
    hold on
 end
 
  %Output all bit layers of watermarked peppers image 
 figure
 suptitle('Watermarked Peppers.tif Bit Layer')
  for kk = 1:8
    bp = bitget(peppers_wm, kk); 
    subplot(2,4,kk)
    imshow(bp, []);
    title(['Layer ' num2str(kk)])
    hold on
  end
 
%swapping half of the water marked images 
figure
title('Altered Image Half and Half')
[row col]=size(peppers_wm);
sc=zeros(row, col);
sc(1:row/2,:)=peppers_wm(1:row/2,:);
sc(row/2+1:row,:)=baboon_wm(row/2+1:row,:);
sc=uint8(sc);
imshow(sc) 
hold on

  %Output all bit layers of altered watermarked peppers image
  %Watermark is preserved
 figure 
  suptitle('Altered Image Bit Layers')
 for m = 1:8
    test = bitget(sc, m); 
    subplot(2,4,m)
    imshow(test, []);
    title(['Layer ' num2str(m)])
    hold on
 end


