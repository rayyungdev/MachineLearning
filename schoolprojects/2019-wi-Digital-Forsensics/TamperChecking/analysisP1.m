function [pixels] = analysisP1(pixels)
[count, bin]=imhist(pixels);
subplot(1,2,1);
imshow(pixels);

subplot(1,2,2);
bar(bin, count);
title('Distribution');
end

