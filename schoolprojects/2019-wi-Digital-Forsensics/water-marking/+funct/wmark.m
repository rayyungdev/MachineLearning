function [test] = wmark(opixel, wpixel, key)
opixel=double(opixel);
rng(key);
lup=rand(1,256) > .5;
wpixel=bitget(wpixel, 8);
[row, col] = size(opixel);
test=zeros(row,col);
for i=1:row
    for k=1:col
        pixval=opixel(i,k)+1; %pixel value from 0-255
        wmarkpix=wpixel(i,k); %pixel value from 0-255
        if (lup(pixval)~= wmarkpix) %compare the watermarkpix value to the lookup table of original
            dif=find(lup==wmarkpix); %find the closest one
            [~,difv]=min(abs(dif-pixval)); %find the index
            try
                test(i,k)=dif(difv)-1; %change the original pixel value to the closest value in the array dif AND SUBTRACT A FUCKING 1 BECAUSE ITS CURRENTLY BETWEEN 1-256
            catch
                fprintf("Error \n")
                
            end
        else
            test(i,k)=pixval-1;
        end
    end
end
test=uint8(test);
end