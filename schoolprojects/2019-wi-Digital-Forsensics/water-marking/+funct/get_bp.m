%Assignment 3

%Write a Matlab function that extracts and displays a user speci?ed bit plane from an image.
function [bp] = get_bp(image, bitplane)
original_image = imread(image);
bp = bitget(original_image, bitplane); 
imshow(bp, []);
end
