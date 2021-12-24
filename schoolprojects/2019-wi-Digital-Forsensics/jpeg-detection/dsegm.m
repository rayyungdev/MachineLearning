function [fmat] = dsegm(seg, exrow, excol)
    fmat=zeros(exrow, excol);
    count=1;
    for k=1:8:exrow
        for m=1:8:excol
            fmat(k:k+7,m:m+7)=seg(:,:,count);
            count=count+1;
        end
    end
end

