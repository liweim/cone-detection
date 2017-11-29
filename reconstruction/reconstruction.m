clear;
clc;
close all;

% Load the stereoParameters object.
load('stereoParams_zed_accurate.mat');

% Visualize camera extrinsics.
%showExtrinsics(stereoParams);

f = 700;
d = 120;

tic;
frame = imread('ZED/stereo/30.png');
frame_size = size(frame);
width = frame_size(2);
frameLeft = frame(:, 1 : width/2, :);
frameRight = frame(:, width/2 +1: width, :);

[frameLeftRect, frameRightRect] = rectifyStereoImages(frameLeft, frameRight, stereoParams);

%imtool(stereoAnaglyph(frameLeftRect, frameRightRect));

coneDisparity = [163.5, 102, 58.5];
coneRealDistance = [0.5, 0.8, 1.4];
for i = 1:3
    coneDepth(i) = f * d / coneDisparity(i);
end
coneDepth;
toc
% frameLeftGray  = rgb2gray(frameLeftRect);
% frameRightGray = rgb2gray(frameRightRect);
% 
% maxRange = 160;
% blockSize = 17;
% % while(1)
% %     maxRange = input('input maxRange: ');
% %     blockSize = input('input blockSize: ');
% %     blockSize = blockSize + 2
% %     maxRange = maxRange + 16
%     disparityRange = [0 maxRange]; %[0 80]
%     disparityMap = disparity(frameLeftGray, frameRightGray, 'BlockSize', blockSize, 'DisparityRange', disparityRange);
%     disparityMap = medfilt2(disparityMap);
%     figure(1);
%     imshow(disparityMap, disparityRange);
%     
%     title(maxRange);
%     colormap jet
%     colorbar
% % end
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
%         if points3D(i,j,3)<10
%             points3D2(i,j,:) = points3D(i,j,:);
%         end
%     end
% end
% 
% ptCloud = pointCloud(points3D2, 'Color', frameLeftRect);
% 
% % Create a streaming point cloud viewer
% player3D = pcplayer([-1, 1], [-1, 1], [0, 5], 'VerticalAxis', 'y', ...
%     'VerticalAxisDir', 'down');
% % Visualize the point cloud
% view(player3D, ptCloud);
