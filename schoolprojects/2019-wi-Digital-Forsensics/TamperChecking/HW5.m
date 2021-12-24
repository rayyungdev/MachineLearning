%% Part 1.1
figure(1); %A little contrast enhanced
P1=imread('imageCE1.tif');
P1=analysisP1(P1);
figure(2);
P2=imread('imageCE2.tif');
P2=analysisP1(P2);
figure(3); %Definitly contrast enhanced
P3=imread('imageCE3.tif');
P3=analysisP1(P3); %
figure(4)
P4=imread('imageCE4.tif');
P4=analysisP1(P4);
%% Part 1.2
UA1=imread('unaltIm1.tif');
UA2=imread('unaltIm2.tif');
UA3=imread('unaltIm3.tif');

UAN1=GammaCorrection(UA1, .7, 0);
figure(5) %Original
AnA1=analysisP1(UA1);
figure(6) %Gamma=0.7 == Lighter, histogram shifts more towards 255, which makes sense
An1=analysisP1(UAN1);
UAN11=GammaCorrection(UA1, 1.3, 0);
figure(7) %Gamma=1.3 == Darker, so the histogram shifts more to 0, and is leftwards leaning. 
An2=analysisP1(UAN11);

UAN2=GammaCorrection(UA2, .7, 0);
figure(8)
AnA2=analysisP1(UA2);
figure(9)
An3=analysisP1(UAN2);
UAN21=GammaCorrection(UA2, 1.3, 0);
figure(10)
An4=analysisP1(UAN21);

UAN3=GammaCorrection(UA3, .7, 0);
figure(11)
AnA3=analysisP1(UA3);
figure(12);
An5=analysisP1(UAN3);
UAN31=GammaCorrection(UA3, 1.3, 0);
figure(13)
An4=analysisP1(UAN31);

%% Part 1.3
figure(14)
An5=imread('imageCE5.tif');
An5=analysisP1(An5); %Gamma is less than 1. 

%% Questions
%{
Based off of your knowledge of contrast enhancement fingerprints, which of these images
are likely to have been contrast enhanced? 
    Looking at the histograms, imageCE3 is the most contrasted enhanced and
    imageCE1 is a little contrast enahced. You can tell because their
    histograms are more spread and also have gaps within the histogram
    which happens when images are contrasted. imageCE3 is used locally
    expansive contrasting while imageCE1 used locally contracting
    contrasting whic you can tell, since imageCE3 has many caps while
    imageCE1 has several random peaks. 
What influence does the unaltered image’s pixel value histogram have on the location and amplitude of the
contrast enhancement fingerprints in each of the gamma corrected images. What influence does the contrast
enhancement mapping (in this case specified by the value of ?) have on the location and amplitude of the
contrast enhancement fingerprints in each of the gamma corrected images. 
    The original contrast of an image already has predefined peaks. This
    can be seen in figures 5, 8, and 11. If the image is already dark, then
    the histogram leans towards the left and has higher amplitudes towards
    the left and vice-versa.
    First, when gamma is less than 1, the picture gets brighter and the
    histogram values shift towards the right. If a picture is already
    bright, then the peaks are significantly more frequent in the 255
    range. However, if the picture is dark, the histogram's spread is
    expanded and peaks occure more frequently but is less in value. This is
    evident in figures 11, 12, and 13.
Examine its pixel value histogram and identify which regions of the contrast enhancement mapping are locally expansive and which are locally contractive. Based off of this information, is ? greater than
1 or less than 1?
    Looking at figure 14, you can tell that the image is locally contracted
    at values 120 and above and locally expansive at values below 120. This
    is seen in the figure because the values are spaced out more for
    locally expansive values while locally contracted values have more
    peaks. The gamma is definitely less than one, since the amplitudes are
    towards the right to 255, showing that the image is brightened. 
%}