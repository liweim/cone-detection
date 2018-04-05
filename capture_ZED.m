clear all;close all;clc;

% Open the ZED
zed = webcam('ZED');
% Set video resolution
%zed.Resolution = zed.AvailableResolutions{1};
% Get image size
%[height width channels] = size(snapshot(zed));
%%
% Create Figure and wait for keyboard interrupt to quit
f = figure('name','ZED camera','keypressfcn','close','windowstyle','modal');
ok = 1;
n=0;
% Start loop
while n < 300 && ok
      n=n+1;
      % Capture the current image
      img = snapshot(zed);

      % Split the side by side image image into two images
%       image_left = img(:, 1 : width/2, :);
%       image_right = img(:, width/2 +1: width, :);
% %       if mod(n,10) == 0
%           imwrite(image_left,strcat('left/',num2str(n),'.png'));
%           imwrite(image_right,strcat('right/',num2str(n),'.png'));
% %       end
%       % Display the left and right images
%       subplot(1,2,1);
%       imshow(image_left);
%       title('Image Left');
%       subplot(1,2,2);
%       imshow(image_right);
%       title('Image Right');

          imwrite(img,strcat('annotations/rainy2/',num2str(n),'.png'));
          imshow(img);
        
      drawnow;

      % Check for interrupts
      ok = ishandle(f);
  end

  % close the camera instance
  clear cam