import numpy as np
import cv2
import os
from os.path import join
import matplotlib.pyplot as plt

mtx_left = np.array([[699.8341, 0, 670.6991], [0, 699.9473, 327.9933], [0, 0, 1]])
dist_left = np.array([-0.1708, 0.0267, 0, 0, 0])
mtx_right = np.array([[702.2891, 0, 667.0359], [0, 701.5237, 358.7018], [0, 0, 1]])
dist_right = np.array([-0.1733, 0.0275, 0, 0, 0])

R = np.array([[0.9998, -0.0016, -0.0215], [0.0016, 1, -0.0021], [0.0215, 0.0020, 0.9998]])
R = np.transpose(R)
T = np.array([-119.1632, 0.2062, 0.0252])

def callback(x):
    pass

def trackbar(left, right):
    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('numDisparities','disparity',1,30,callback)
    cv2.createTrackbar('blockSize','disparity',0,10,callback)
    cv2.createTrackbar('uniquenessRatio','disparity',0,20,callback)

    r, c = left.shape[:2]
    '''
    cv2.namedWindow('left', cv2.WINDOW_NORMAL)
    cv2.imshow('left', left)
    cv2.waitKey(0)
    '''
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        numDisparities = cv2.getTrackbarPos('numDisparities','disparity')
        blockSize = cv2.getTrackbarPos('blockSize','disparity')
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disparity')

        numDisparities = numDisparities * 16
        blockSize = 5 + blockSize * 2
        stereo = cv2.StereoSGBM_create(numDisparities = numDisparities, blockSize = blockSize, uniquenessRatio = uniquenessRatio)
        #stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
        disparity = stereo.compute(left, right)
        #depth = 34571 * 726 / disparity[460, 819 + numDisparities]
        print(numDisparities, blockSize, uniquenessRatio)
        #disparity = cv2.inRange(disparity, 0, 100)
        min = disparity.min()
        max = disparity.max()
        disparity = np.uint8(255 * (disparity - min) / (max - min))
        cv2.imshow('disparity', disparity)

    cv2.destroyAllWindows()
    return numDisparities, blockSize, uniquenessRatio

def reconstruction(img_path):
    img = cv2.imread(img_path)
    row, col = img.shape[:2]

    left = img[:, :int(col/2)]
    right = img[:, int(col/2):]

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1=mtx_left,
        distCoeffs1=dist_left,
        cameraMatrix2=mtx_right,
        distCoeffs2=dist_right,
        imageSize=(1280, 720),
        R=R,
        T=T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0)

    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix=mtx_left,
        distCoeffs=dist_left,
        R=R1,
        newCameraMatrix=P1,
        size=(1280, 720),
        m1type=cv2.CV_32FC1)

    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix=mtx_right,
        distCoeffs=dist_right,
        R=R2,
        newCameraMatrix=P2,
        size=(1280, 720),
        m1type=cv2.CV_32FC1)


    left_rect = cv2.remap(left, map1x, map1y, cv2.INTER_LINEAR)
    right_rect = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)
    cv2.imshow('rectify', left_rect+right_rect)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    left_rect = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_rect = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    #numDisparities, blockSize, uniquenessRatio = trackbar(left_rect, right_rect)

    numDisparities = 224
    blockSize = 21
    uniquenessRatio = 3
    #stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    stereo = cv2.StereoSGBM_create(numDisparities = numDisparities, blockSize = blockSize, uniquenessRatio = uniquenessRatio)
    disparity = stereo.compute(left_rect, right_rect)

    plt.imshow(disparity, cmap='gray')
    plt.show()


if __name__=='__main__':
    reconstruction('ZED/stereo/2.png')
