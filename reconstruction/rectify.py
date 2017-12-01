import numpy as np
import cv2
import os
from os.path import join
import matplotlib.pyplot as plt
import argparse

mtx_left = np.array([[699.8341, 0, 670.6991], [0, 699.9473, 327.9933], [0, 0, 1]])
dist_left = np.array([-0.1708, 0.0267, 0, 0, 0])
mtx_right = np.array([[702.2891, 0, 667.0359], [0, 701.5237, 358.7018], [0, 0, 1]])
dist_right = np.array([-0.1733, 0.0275, 0, 0, 0])

R = np.array([[0.9998, -0.0016, -0.0215], [0.0016, 1, -0.0021], [0.0215, 0.0020, 0.9998]])
R = np.transpose(R)
T = np.array([-119.1632, 0.2062, 0.0252])

F = 700
d = 120

def callback(x):
    pass

def trackbar(left, right):
    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('numDisparities','disparity',1,30,callback)
    cv2.createTrackbar('blockSize','disparity',0,10,callback)
    cv2.createTrackbar('uniquenessRatio','disparity',0,20,callback)

    r, c = left.shape[:2]

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        numDisparities = cv2.getTrackbarPos('numDisparities','disparity')
        blockSize = cv2.getTrackbarPos('blockSize','disparity')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disparity')

        numDisparities = numDisparities * 16
        blockSize = 5 + blockSize * 2
        #stereo = cv2.StereoSGBM_create(numDisparities = numDisparities, blockSize = blockSize, uniquenessRatio = uniquenessRatio)
        #stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

        window_size = 3
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = numDisparities,
            blockSize = blockSize,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )

        disparity = stereo.compute(left, right)
        depth_map = F * d / disparity
        print(numDisparities, blockSize, uniquenessRatio)
        #depth_map = cv2.inRange(depth_map, 0, 1000)
        result = cv2.hconcat((disparity, depth_map))
        cv2.imshow('disparity', result)

    cv2.destroyAllWindows()
    return numDisparities, blockSize, uniquenessRatio

def rectify(img_path):
    basename = os.path.split(img_path)[1]
    dirname = os.path.split(img_path)[0]
    dirname = os.path.split(dirname)[0]
    num = os.path.split(dirname)[1]
    img = cv2.imread(img_path)
    row, col = img.shape[:2]
    col = int(col/2)
    left = img[:, :col]
    right = img[:, col:]
    row = 1280
    col = 720
    left = cv2.resize(left, (row, col))
    right = cv2.resize(right, (row, col))

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=mtx_left,
        distCoeffs1=dist_left,
        cameraMatrix2=mtx_right,
        distCoeffs2=dist_right,
        imageSize=(row, col),
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix=mtx_left,
        distCoeffs=dist_left,
        R=R1,
        newCameraMatrix=P1,
        size=(row, col),
        m1type=cv2.CV_32FC1)

    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix=mtx_right,
        distCoeffs=dist_right,
        R=R2,
        newCameraMatrix=P2,
        size=(row, col),
        m1type=cv2.CV_32FC1)

    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    left_rect = cv2.resize(left_rect, (320, 180))
    left_rect = cv2.resize(left_rect, (320, 180))

    left_save_path = join('ZED', num, 'rectify', 'left', basename)
    right_save_path = join('ZED', num, 'rectify', 'right', basename)
    cv2.imwrite(left_save_path, left_rect)
    cv2.imwrite(right_save_path, right_rect)
    return left_rect, right_rect

'''
rectify = left_rect + right_rect

plt.imshow(rectify[:,:,::-1])
pts = np.array(pl.ginput(100, timeout = 10^10))
num_pt = len(pts)
print(num_pt)
if num_pt % 2 == 0:
    coneDisparity = np.zeros(int(num_pt/2))
    for i in range(0, num_pt, 2):
        n = int(i/2)
        coneDisparity[n] = np.linalg.norm(pts[i] - pts[i+1])

    coneDepth = F * d / coneDisparity
    print(coneDepth)
else:
    print('number of points should be even')
'''
'''
left_rect = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
right_rect = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

#numDisparities, blockSize, uniquenessRatio = trackbar(left_rect, right_rect)
numDisparities = 224
blockSize = 21
uniquenessRatio = 3
stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
#stereo = cv2.StereoSGBM_create(numDisparities = numDisparities, blockSize = blockSize, uniquenessRatio = uniquenessRatio)
disparity = stereo.compute(left_rect, right_rect)
depth_map = np.zeros(disparity.shape)
for r in range(row):
    for c in range(col):
        if disparity[r, c] > 0:
            depth_map[r, c] = F * d / disparity[r, c]
#depth_map = cv2.inRange(depth_map, 0, 1000)
f, ax = plt.subplots(2, 1)
ax[0].imshow(disparity, cmap='gray')
ax[1].imshow(depth_map, cmap='gray')

plt.show()
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    args = parser.parse_args()

    rectify(img_path = args.img_path)
