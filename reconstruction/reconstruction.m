clear;
clc;
close all;

% Load the stereoParameters object.
load('stereoParams_zed_accurate.mat');

% Visualize camera extrinsics.
%showExtrinsics(stereoParams);

f = 700;
d = 120;

frameLeft = imread('calibration/left_zed/10.png');
frameRight = imread('calibration/right_zed/10.png');

[frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, stereoParams);

imtool(stereoAnaglyph(frameLeftRect, frameRightRect));

disparity = 30;
depth = f * d / disparity

% frameLeftGray  = rgb2gray(frameLeftRect);
% frameRightGray = rgb2gray(frameRightRect);

% maxRange = 384;
% % blockSize = 3;
% %while(1)
%     %maxRange = input('input maxRange: ');
%     %blockSize = input('input blockSize: ');
% %     blockSize = blockSize + 2;
% %     maxRange = maxRange + 16;
%     disparityRange = [0 maxRange]; %[0 80]
%     disparityMap = disparity(frameLeftGray, frameRightGray, 'BlockSize', 15, 'DisparityRange', disparityRange);
%     disparityMap = medfilt2(disparityMap);
%     figure(1);
%     imshow(disparityMap, disparityRange);
%     
%     title(maxRange);
%     colormap jet
%     colorbar
% %end
% 
% points3D = reconstructScene(disparityMap, stereoParams);
% 
% 
% % Convert to meters and create a pointCloud object
% points3D = points3D ./ 1000;
% [r,c,n] = size(points3D);
% points3D2 = zeros([r,c,n]);
% for i = 1:r
%     for j = 1:c
%         if points3D(i,j,3)<1
%             points3D2(i,j,:) = points3D(i,j,:);
%         end
%     end
% end
% 
% ptCloud = pointCloud(points3D2, 'Color', frameLeftRect);
% 
% % Create a streaming point cloud viewer
% player3D = pcplayer([-1, 1], [-1, 1], [0, 1], 'VerticalAxis', 'y', ...
%     'VerticalAxisDir', 'down');
% % Visualize the point cloud
% view(player3D, ptCloud);
