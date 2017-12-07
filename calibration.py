import numpy as np
import cv2
import glob
import os
from os.path import join
import matplotlib.pyplot as plt

def cropHorizontal(image, crop_shape):
    r, c = image.shape[:2]
    crop_r, crop_c = crop_shape
    return image[int((r-crop_r)/2):int(crop_r+(r-crop_r)/2),int((c-crop_c)/2):int(crop_c+(c-crop_c)/2)]

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
