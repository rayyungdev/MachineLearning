function showFreqPmap(pmap)

freqPmap= abs(fft2(pmap-mean(mean(pmap))));
freqShiftPmap= fftshift(freqPmap);


blurFilt= fspecial('gaussian', 7, 10);
blurredFreqShiftPmap= filter2(blurFilt,freqShiftPmap);

mapMax= max(max(blurredFreqShiftPmap));
mapMin= min(min(blurredFreqShiftPmap));

[mapRows mapCols]= size(blurredFreqShiftPmap);


% if (mapRows/2) ~= floor(mapRows/2)
%     mapRows= mapRows-1;
% end
% 
% if (mapCols/2) ~= floor(mapCols/2)
%     mapCols= mapCols-1;
% end

% quarterRowMat=[1:mapRows/2]'*ones(1,mapCols/2)./(mapRows/2);
% quarterColMat=ones(mapRows/2,1)*[1:mapCols/2]./(mapCols/2);
% quarterMat= (quarterRowMat.^2 + quarterColMat.^2).^.5;


quarterRowMat=[1:floor(mapRows/2)]'*ones(1,floor(mapCols/2))./floor(mapRows/2);
quarterColMat=ones(floor(mapRows/2),1)*[1:floor(mapCols/2)]./floor(mapCols/2);
quarterMat= (quarterRowMat.^2 + quarterColMat.^2).^.5;

radius= .125;%.15;%20;
quarterMask= quarterMat > radius;
Mask(1:floor(mapRows/2),1:floor(mapCols/2))= quarterMask(end:-1:1,end:-1:1);
Mask(1:floor(mapRows/2),round(mapCols/2+1):mapCols)= quarterMask(end:-1:1,1:end);
Mask(round(mapRows/2+1):mapRows,1:floor(mapCols/2))= quarterMask(1:end,end:-1:1);
Mask(round(mapRows/2+1):mapRows,round(mapCols/2+1):mapCols)= quarterMask;

% Mask(1:mapRows/2,1:mapCols/2)= quarterMask(end:-1:1,end:-1:1);
% Mask(1:mapRows/2,(mapCols/2+1):mapCols)= quarterMask(end:-1:1,1:end);
% Mask((mapRows/2+1):mapRows,1:mapCols/2)= quarterMask(1:end,end:-1:1);
% Mask((mapRows/2+1):mapRows,(mapCols/2+1):mapCols)= quarterMask;




blurredFreqShiftPmap= abs(blurredFreqShiftPmap.*Mask);
blurredFreqShiftPmap= (abs(blurredFreqShiftPmap- mapMin)/mapMax).^2.5;%1.5;

%blurredFreqShiftPmap= ((blurredFreqShiftPmap- mapMin)/mapMax).^0.8;


%imagesc(freqShiftPmap);
imagesc(blurredFreqShiftPmap);
colormap(gray);
set(gca,'XTick',[])
set(gca,'YTick',[])