function [seg, row, col] = segm(Pixels)
    [row, col] = size(Pixels);
    counter=(row*col)/64;
    seg=zeros(8,8,counter);
    count=1;
    for k=1:8:row
        for m=1:8:col
            seg(:,:,count)=Pixels(k:k+7, m:m+7);
            count=count+1;
        end
    end
end

