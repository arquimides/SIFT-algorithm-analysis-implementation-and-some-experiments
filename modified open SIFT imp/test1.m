% 
%   Test 1: Generating and visualizing keypoints localizations and descriptions for
%   user selected input image.

clear
tic
% Some parameters as suggested in the Lowe's original paper "Distinctive Image Features from Scale-Invariant Keypoints"
options.octavesNum = 4;
options.scalesNum = 5;
options.initialSigma = 1.6; % See section 3.3
options.k = sqrt(2);
options.doubleSizeInitialImage = 'TRUE';

% Loading the input image
name_path1 = uigetfile('*.png','Select the input image');
[image1, des1,loc1] = sift(strcat('images/',name_path1),options);

% Visualizing keypoints
showkeys(image1,loc1);
toc