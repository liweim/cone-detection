import numpy as np
import cv2
import glob
import os
from os.path import join
import matplotlib.pyplot as plt
'''
mtx_left = np.array([[723.7591, 0, 672.5802], [0, 721.7413, 351.4963], [0, 0, 1]])
dist_left = np.array([-0.3393, 0.1499, -0.0360, 0, 0])
mtx_right = np.array([[725.2446, 0, 687.6786], [0, 723.3846, 325.4647], [0, 0, 1]])
dist_right = np.array([-0.326, 0.1279, -0.0253, 0, 0])
'''
'''
mtx_left = np.array([[726.9435, 0, 671.3295], [0, 725.0371, 348.9871], [0, 0, 1]])
dist_left = np.array([-0.3422, 0.1556, -0.0389, 0, 0])
mtx_right = np.array([[722.2726, 0, 689.1013], [0, 720.1296, 324.9532], [0, 0, 1]])
dist_right = np.array([-0.3446, 0.1647, -0.0464, 0, 0])
'''
'''
mtx_left = np.array([[726.2317, 0, 672.0512], [0, 723.9083, 354.7856], [0, 0, 1]])
dist_left = np.array([-0.3381, 0.1229, 0, 0, 0])
mtx_right = np.array([[731.1806, 0, 688.8168], [0, 730.3499, 328.6535], [0, 0, 1]])
dist_right = np.array([-0.3049, 0.0804, 0, 0, 0])

R = np.array([[0.9991, 0.0429, -0.0028], [-0.0429, 0.9991, 0.0026], [0.0029, -0.0024, 1]])
#R = np.transpose(R)
T = np.array([-99.8791, -1.7033, -3.2390])
'''
mtx_left = np.array([[699.2363, 0, 670.4346], [0, 699.1530, 327.0556], [0, 0, 1]])
dist_left = np.array([-0.1686, 0.0230, 0, 0, 0])
mtx_right = np.array([[701.8448, 0, 666.6404], [0, 701.0176, 357.8100], [0, 0, 1]])
dist_right = np.array([-0.1734, 0.0275, 0, 0, 0])

R = np.array([[0.9998, -0.0018, -0.0220], [0.0018, 1, -0.0019], [0.0220, 0.0019, 0.9998]])
R = np.transpose(R)
T = np.array([-119.3038, -0.2544, 0.1172])

def cropHorizontal(image, crop_shape):
    r, c = image.shape[:2]
    crop_r, crop_c = crop_shape
    return image[int((r-crop_r)/2):int(crop_r+(r-crop_r)/2),int((c-crop_c)/2):int(crop_c+(c-crop_c)/2)]

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
        #stereo = cv2.StereoSGBM_create(numDisparities = numDisparities, blockSize = blockSize, uniquenessRatio = uniquenessRatio)
        stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
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

def mono_calibration(img_folder_path):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(join(img_folder_path, '*.png'))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners)
            '''
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(500)
            '''

    cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    mean_error = 0
    tot_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error

    print(mtx, dist)
    print("total error: ", mean_error/len(objpoints))
    basename = os.path.split(img_folder_path)[1]
    save_path = basename + '_mtx.txt'
    np.savetxt(save_path, mtx)
    save_path = basename + '_dist.txt'
    np.savetxt(save_path, dist)
    return ret, mtx, dist, rvecs, tvecs

def undistort_img(img_folder_path, img_path, crop_shape):
    paths = ['left', 'right']
    mtxs = [mtx_left, mtx_right]
    dists = [dist_left, dist_right]
    for path, mtx, dist in zip(paths, mtxs, dists):
        img = cv2.imread(join(img_folder_path, path, img_path))
        #img = cropHorizontal(img, crop_shape)
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

        # undistort
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        cv2.imshow(path, dst)

        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(join('result',path+'_'+img_path), dst)

    cv2.waitKey(0)

def reconstruction(img_path):
    left_img_path = 'calibration/left_zed/'+img_path
    right_img_path = 'calibration/right_zed/'+img_path

    left = cv2.imread(left_img_path)
    right = cv2.imread(right_img_path)

    left = cv2.resize(left, (1280, 720))
    right = cv2.resize(right, (1280, 720))

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

    left_rect = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
    right_rect = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)

    '''
    left_img_path = 'right.png'
    right_img_path = 'left.png'

    left = cv2.imread(left_img_path)
    right = cv2.imread(right_img_path)

    left_rect = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_rect = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    '''
    #numDisparities, blockSize, uniquenessRatio = trackbar(left_rect, right_rect)
    #stereo = cv2.StereoSGBM_create(numDisparities = numDisparities, blockSize = blockSize, uniquenessRatio = uniquenessRatio)
    numDisparities = 384
    blockSize = 15
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(left_rect, right_rect)
    min = disparity.min()
    max = disparity.max()
    disparity = np.uint8(255 * (disparity - min) / (max - min))
    cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
    cv2.imshow('disparity', disparity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    #mono_calibration('calibration/left')
    #undistort_img(img_folder_path = 'calibration', img_path = '40.png', crop_shape = [540, 960])
    reconstruction('1.png')
