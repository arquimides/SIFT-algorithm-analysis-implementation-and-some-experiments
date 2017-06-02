function [ image, descrs, locs ] = sift( input_img_path, options )
% Function: Get sift features and descriptors
%   Input parameters:
%     input_img_path: the image file path name.
%     options: some required parameters for the algorithm
%
%   Returned:
%     image: the image array in double format
%     descrs: a K-by-128 matrix, where each row gives an invariant descriptor for one of the K keypoints. 
%                  The descriptor is a vector of 128 values normalized to unit length.
%     locs:   K-by-4 matrix, in which each row has the 4 values for a keypoint location (row, column, scale, orientation).
%                  The orientation is in the range [-PI, PI] radians.

%% Preprocesing the image and initializing some parameters
global gauss_pyr;
global dog_pyr;
global init_sigma;
global octvs;
global intvls;
global ddata_array;
global features;

input_img = imread(input_img_path);
image = input_img;

if(size(input_img,3)==3) % Check if the image is in color
    input_img = rgb2gray(input_img); % Convert to gray scale
end

input_img = im2double(input_img);

% initial sigma
init_sigma = options.initialSigma;
% number of intervals per octave
intvls = 3;
s = intvls;
k = 2^(1/s);
sigma = ones(1,s+3);
sigma(1) = init_sigma;
sigma(2) = init_sigma*sqrt(k*k-1);
for i = 3:s+3
    sigma(i) = sigma(i-1)*k;
end
%Double the image size using a linear interpolation
if(strcmp(options.doubleSizeInitialImage,'TRUE'))
    input_img = imresize(input_img,2,'bilinear');
end
% assume the original image has a blur of sigma = 0.5
input_img = gaussian(input_img,sqrt(init_sigma^2-0.5^2*4));

% smallest dimension of top level is about 8 pixels
octvs = floor(log( min(size(input_img)) )/log(2) - 2);

%% Step 1(The Scale-space): Construct the scale space
% gaussian pyramid
[img_height,img_width] =  size(input_img);
gauss_pyr = cell(octvs,1);
% set image size
gimg_size = zeros(octvs,2);
gimg_size(1,:) = [img_height,img_width];
for i = 1:octvs
    if (i~=1)
        gimg_size(i,:) = [round(size(gauss_pyr{i-1},1)/2),round(size(gauss_pyr{i-1},2)/2)];
    end
    gauss_pyr{i} = zeros( gimg_size(i,1),gimg_size(i,2),s+3 );
end
for i = 1:octvs
    for j = 1:s+3
        if (i==1 && j==1)
            gauss_pyr{i}(:,:,j) = input_img;
        % downsample for the first image in an octave, from the s+1 image
        % in previous octave.
        elseif (j==1)
            gauss_pyr{i}(:,:,j) = imresize(gauss_pyr{i-1}(:,:,s+1),0.5);
        else
            gauss_pyr{i}(:,:,j) = gaussian(gauss_pyr{i}(:,:,j-1),sigma(j));
        end
    end
end

%% Step 2 (LoG approximations): Calculate a LoG approximation using DoG
dog_pyr = cell(octvs,1);
for i = 1:octvs
    dog_pyr{i} = zeros(gimg_size(i,1),gimg_size(i,2),s+2);
    for j = 1:s+2
    dog_pyr{i}(:,:,j) = gauss_pyr{i}(:,:,j+1) - gauss_pyr{i}(:,:,j);
    end
end
%Uncoment the following lines to see the DoGs
%for i = 3:size(dog_pyr,1)
%    for j = 1:size(dog_pyr{i},3)
%        imshow(dog_pyr{i}(:,:,j));
%    end
%end

%% Step 3 (Finding Keypoints) and Step 4 (Filtering Keypoints): Finding candidate keypoints using the DoG and carefully filter some of them
% width of border in which to ignore keypoints
img_border = 5;
% maximum steps of keypoint interpolation
max_interp_steps = 5;
% low threshold on feature contrast
contr_thr = 0.03;
% high threshold on feature ratio of principal curvatures
curv_thr = 10;
prelim_contr_thr = 0.5*contr_thr/intvls;
ddata_array = struct('x',0,'y',0,'octv',0,'intvl',0,'x_hat',[0,0,0],'scl_octv',0);
ddata_index = 1;
for i = 1:octvs
    [height, width] = size(dog_pyr{i}(:,:,1));
    % find extrema in middle intvls
    for j = 2:s+1
        dog_imgs = dog_pyr{i};
        dog_img = dog_imgs(:,:,j);
        for x = img_border+1:height-img_border
            for y = img_border+1:width-img_border
                % preliminary check on contrast
                if(abs(dog_img(x,y)) > prelim_contr_thr)
                    % check 26 neighboring pixels
                    if(isExtremum(j,x,y))
                        ddata = interpLocation(dog_imgs,height,width,i,j,x,y,img_border,contr_thr,max_interp_steps);
                        if(~isempty(ddata))
                            if(~isEdgeLike(dog_img,ddata.x,ddata.y,curv_thr))
                                 ddata_array(ddata_index) = ddata;
                                 ddata_index = ddata_index + 1;
                            end
                        end
                    end
                end
            end
        end
         
    end
    %drawMaxAndMin(dog_img, ddata_array);
end

function [ flag ] = isExtremum( intvl, x, y)
% Function: Find Extrema in 26 neighboring pixels
    value = dog_imgs(x,y,intvl);
    block = dog_imgs(x-1:x+1,y-1:y+1,intvl-1:intvl+1);
    if ( value > 0 && value == max(block(:)) )
        flag = 1;
    elseif ( value == min(block(:)) )
        flag = 1;
    else
        flag = 0;
    end
end

%%  Step 5 (Keypoint orientations): One or more orientations are assigned to each keypoint location based on local image gradient directions. 
%                                   All future operations are performed on image data that has been transformed relative to the assigned orientation,
%                                   scale, and location for each feature, thereby providing invariance to these transformations.
% number of detected points
n = size(ddata_array,2);
% determines gaussian sigma for orientation assignment
ori_sig_factr = 1.5;
% number of bins in histogram
ori_hist_bins = 36;
% orientation magnitude relative to max that results in new feature
ori_peak_ratio = 0.8;
% array of features
features = struct('ddata_index',0,'x',0,'y',0,'scl',0,'ori',0,'descr',[]);
feat_index = 1;
for i = 1:n
    ddata = ddata_array(i);
    ori_sigma = ori_sig_factr * ddata.scl_octv;
    % generate a histogram for the gradient distribution around a keypoint
    hist = oriHist(gauss_pyr{ddata.octv}(:,:,ddata.intvl),ddata.x,ddata.y,ori_hist_bins,round(3*ori_sigma),ori_sigma);
    for j = 1:2
        smoothOriHist(hist,ori_hist_bins);
    end
    % generate feature from ddata and orientation hist peak
    % add orientations greater than or equal to 80% of the largest orientation magnitude
    feat_index = addOriFeatures(i,feat_index,ddata,hist,ori_hist_bins,ori_peak_ratio);
    %Uncomment to plot de hist at this point
    %plotHist(hist);
end

%% Step 6 (Keypoint descriptor) : The local image gradients are measured at the selected scale in the region around each keypoint. 
%                                 These are transformed into a representation that allows for significant levels of local shape distortion and 
%                                 change in illumination.
% number of features
n = size(features,2);
disp(strcat('Number of keypoints : ', num2str(n)));
% width of 2d array of orientation histograms
descr_hist_d = 4;
% bins per orientation histogram
descr_hist_obins = 8;
% threshold on magnitude of elements of descriptor vector
descr_mag_thr = 0.2;
descr_length = descr_hist_d*descr_hist_d*descr_hist_obins;
local_features = features;
local_ddata_array = ddata_array;
local_gauss_pyr = gauss_pyr;
clear features;
clear ddata_array;
clear gauss_pyr;
clear dog_pyr;
parfor feat_index = 1:n
    feat = local_features(feat_index);
    ddata = local_ddata_array(feat.ddata_index);
    gauss_img = local_gauss_pyr{ddata.octv}(:,:,ddata.intvl);
% computes the 2D array of orientation histograms that form the feature descriptor
    hist_width = 3*ddata.scl_octv;
    radius = round( hist_width * (descr_hist_d + 1) * sqrt(2) / 2 );
    feat_ori = feat.ori;
    ddata_x = ddata.x;
    ddata_y = ddata.y;
    hist = zeros(1,descr_length);
    for i = -radius:radius
        for j = -radius:radius
            j_rot = j*cos(feat_ori) - i*sin(feat_ori);
            i_rot = j*sin(feat_ori) + i*cos(feat_ori);
            r_bin = i_rot/hist_width + descr_hist_d/2 - 0.5;
            c_bin = j_rot/hist_width + descr_hist_d/2 - 0.5;
            if (r_bin > -1 && r_bin < descr_hist_d && c_bin > -1 && c_bin < descr_hist_d)
                mag_ori = calcGrad(gauss_img,ddata_x+i,ddata_y+j);
                if (mag_ori(1) ~= -1)
                    ori = mag_ori(2);
                    ori = ori - feat_ori;
                    while (ori < 0)
                        ori = ori + 2*pi;
                    end
                    % i think it's theoretically impossible
                    while (ori >= 2*pi)
                        ori = ori - 2*pi;
                        disp('###################what the fuck?###################');
                    end
                    o_bin = ori * descr_hist_obins / (2*pi);
                    w = exp( -(j_rot*j_rot+i_rot*i_rot) / (2*(0.5*descr_hist_d*hist_width)^2) );
                    hist = interpHistEntry(hist,r_bin,c_bin,o_bin,mag_ori(1)*w,descr_hist_d,descr_hist_obins);
                end
            end
        end
    end
    local_features(feat_index) = hist2Descr(feat,hist,descr_mag_thr);
end
% sort the descriptors by descending scale order
features_scl = [local_features.scl];
[~,features_order] = sort(features_scl,'descend');
% return descriptors and locations
descrs = zeros(n,descr_length);
locs = zeros(n,4);
for i = 1:n
    descrs(i,:) = local_features(features_order(i)).descr;
    locs(i,1) = local_features(features_order(i)).x;
    locs(i,2) = local_features(features_order(i)).y;
    locs(i,3) = local_features(features_order(i)).scl;
    locs(i,4) = local_features(features_order(i)).ori;
end

end