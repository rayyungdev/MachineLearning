n=0.25; %threshold
b1=imread('blockArtifacts1.tif');
b2=imread('blockArtifacts2.tif');
b3=imread('blockArtifacts3.tif');
k1=tampcheck(b1) %0.2461 not tampered with
k2=tampcheck(b2) %0.1816 not tampered with
k3=tampcheck(b3) %0.2734 is greater than the threshold and is therefore tampered with. 
%{
Questions
Briefly explain why the histograms of K' and K'' values should be different if an image has been JPEG
compressed.
    If an image has been jpeg compressed, then it has been segmented into
    8x8 pieces, where each segment has its own calculations done so that it
    can be compressed. This results in greater differences between the
    segments, which is why you'll see larger differences between k' (since
    it is the bit values in 1 block), and k'' (which is the bit values in 4
    different blocks). 
%}


