peppers=imread('..\data\peppers.tif'); %set up peppers
bab=imread('..\data\baboon.tif'); %set up baboon
barb=imread('..\data\Barbara.bmp'); %set up watermark

wbarb=logical(bitget(barb,8));
key=435;
wpeppers=funct.wmark(peppers,barb,key); %watermarked peppers
wbab=funct.wmark(bab,barb,key); %watermarked baboon
%%
[ppsnr,~]=psnr(peppers,wpeppers); %PSNR Peppers
[bpsnr,~]=psnr(bab,wbab); %PSNR baboon
%%
ppeppers=funct.wpull(wpeppers,key);
pbab=funct.wpull(wbab,key);
finalpullo=imread('..\data\YMwmkedKey435.tiff');
finalpull=funct.wpull(finalpullo, key);
%%
figure(1)
pplane=logical(bitget(wpeppers, 1));
montage({peppers,wpeppers, pplane, ppeppers}); 

figure(2)
bplane=logical(bitget(wpeppers,1));
montage({bab, wbab, bplane, pbab});

figure(3)
imshowpair(finalpullo, finalpull, 'montage')
%%
[row, col]=size(wpeppers);
sc=zeros(row, col);
sc(1:row/2,:)=wpeppers(1:row/2,:);
sc(row/2+1:row,:)=wbab(row/2+1:row,:);
sc=uint8(sc);
scn=wpull(sc,key);
figure(4)
imshowpair(sc, scn, 'montage');

%{
  As you can see in the two images above, in figure 1 and figure 2, the
modified picture and the watermarked pictures basically the same. You
cannot tell which one is which without looking at the code. 
the noisy image shown on the bottom left of both figure 1 and figure 2 are
the lowest bitplanes, which show nothing but noise.
Furthermore, the PSNR of both the Yeung-Mintzer images are signifantly
higher than the bitplane values. This makes sense because the
Yeung-Mintzer image changes less pixel values, where the LSB image has an
entire bitplane replaced. The PSNR values are shown above (46+)
   
    The Y-m image cannot be recreated when you tamper with it if you do not
have the key. The image shown here uses a key that is one higher than
the original key, and therefore distorts the image completely. However,
if you used the original key with the spliced image you can regain the
watermark. While unreasonable, this type of model is somewhat
susceptible to bruteforcing, in which you can try any possible number of
combinations. This is significantly more difficult and will take significantly 
longer. 

    However, you can recreate the watermark easily in the LSB image, as you
can simply replace the bitplanes back with the watermark image, that or
even just combine the two images together, as they both share the same
bitplane. This can be seen in the image above. 
%}
imshow(barb)
