%Replace N LSB layers from image with N MSB bits of hidden image
%Output modified uint image

function [img]  = LSB(image,hidden,N)
bp = {}; %Create an empty cell array 

for jj = 1:8 
    bpp = funct.get_bp(image, jj); %Obtain bit layers from 1 - 8 of image
    bp{1,jj} = bpp; %populate cell array with bit layers 1 - 8 
end

for ii = 1:N
   hidden_bp = funct.get_bp(hidden, 9-ii); %Obtain N bit layers from hidden image
                                     %Subtract 9 since want MSB (start
                                     %from end)
                                     
                                     
   bp{1,ii} = hidden_bp; %Replace N LSB layer of cell array with N MSB of hidden layer
  
%    %Sanity check that it was replaced
% if bp{1,ii} == hidden_bp
%     disp("True")
% else
%     disp("false")
% end

end


%Reconstruct image by multiplying 2^K * bit layer(K) 
img = uint8(bp{1} + 2*bp{2} + 4*bp{3}  + 8*bp{4} + 16*bp{5} + 32*bp{6} + 64*bp{7} + 128*bp{8});

%Display final image 
imshow(img); 


end


