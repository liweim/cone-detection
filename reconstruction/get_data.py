import numpy as np
import cv2
from matplotlib import pyplot as plt
from os.path import join
import skvideo.io
import skvideo.datasets
import os

def cropHorizontal(image, crop_shape):
    r, c = image.shape[:2]
    crop_r, crop_c = crop_shape
    return image[int((r-crop_r)/2):int(crop_r+(r-crop_r)/2),int((c-crop_c)/2):int(crop_c+(c-crop_c)/2)]

def video2frame(video_folder_path, crop_shape):
    left_video_path = join(video_folder_path, 'left_close.h264')
    right_video_path = join(video_folder_path, 'right_close.h264')
    left_video = skvideo.io.vreader(left_video_path)
    right_video = skvideo.io.vreader(right_video_path)
    n = 0
    m = 0
    for left in left_video:
        n += 1
        if n % 20 == 0:
            left = left[:,:,::-1] #convert to cv2 format
            #left = cropHorizontal(left, crop_shape)
            cv2.imwrite(join(video_folder_path, 'left', str(n)+'.png'), left)
    for right in right_video:
        m += 1
        if m % 20 == 0:
            right = right[:,:,::-1]
            #right = cropHorizontal(right, crop_shape)
            cv2.imwrite(join(video_folder_path, 'right', str(m)+'.png'), right)
    print(n, m)
    return
    '''
    for left, right in zip(left_video, right_video):
        n += 1
        if n % 5 == 0:
            left = left[:,:,::-1] #convert to cv2 format
            cv2.imwrite(join(video_folder_path, 'left', str(n)+'.png'), left)
            right = right[:,:,::-1]
            cv2.imwrite(join(video_folder_path, 'right', str(n)+'.png'), right)
            '''

def slice_img(img_folder_path):
    img_paths = os.listdir(img_folder_path)
    n = 0
    for img_path in img_paths:
        n += 1
        img = cv2.imread(join(img_folder_path, img_path))
        print(join(img_folder_path, img_path))
        r, c = img.shape[:2]
        img_left = img[:, :int(c/2), :]
        img_right = img[:, int(c/2):, :]
        cv2.imwrite(join('calibration', 'left_zed', str(n)+'.png'), img_left)
        cv2.imwrite(join('calibration', 'right_zed', str(n)+'.png'), img_right)


def video2frame_quad(video_path):
    video = skvideo.io.vreader(video_path)
    n = 0
    for frame in video:
        n += 1
        if n % 20 == 0:
            frame = np.array(frame)
            rows, cols, d = frame.shape
            row = int(rows/2)
            col = int(cols/2)
            left = frame[row+4:rows, 0:col+8, :]
            right = frame[row+4:rows, col+16:cols, :]
            left = cv2.resize(left, (1280, 720))
            right = cv2.resize(right, (1280, 720))
            '''
            print(left.shape, right.shape)
            cv2.imshow('left', left)
            cv2.imshow('right', right)
            cv2.waitKey(0)
            return
            '''
            left = left[:,:,::-1] #convert to cv2 format
            cv2.imwrite(join('video2', 'left', str(n)+'.png'), left)
            right = right[:,:,::-1]
            cv2.imwrite(join('video2', 'right', str(n)+'.png'), right)

def reconstruction(img_folder_path):
    left_folder_path = join(img_folder_path, 'left')
    right_folder_path = join(img_folder_path, 'right')
    paths = os.listdir(left_folder_path)
    for path in paths:
        left_img_path = join(left_folder_path, path)
        right_img_path = join(right_folder_path, path)
        left = cv2.imread(left_img_path)
        right = cv2.imread(right_img_path)

        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(left, right)
        plt.imshow(disparity, 'gray')
        plt.show()

        '''
        #Obtainment of the correspondent point with SIFT
        sift = cv2.SIFT()

        ###find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(left,None)
        kp2, des2 = sift.detectAndCompute(right,None)
        return
        ###FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        good = []
        pts1 = []
        pts2 = []

        ###ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)


        pts1 = np.array(pts1)
        pts2 = np.array(pts2)

        #Computation of the fundamental matrix
        F,mask= cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)


        # Obtainment of the rectification matrix and use of the warpPerspective to transform them...
        pts1 = pts1[:,:][mask.ravel()==1]
        pts2 = pts2[:,:][mask.ravel()==1]

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        p1fNew = pts1.reshape((pts1.shape[0] * 2, 1))
        p2fNew = pts2.reshape((pts2.shape[0] * 2, 1))

        retBool ,rectmat1, rectmat2 = cv2.stereoRectifyUncalibrated(p1fNew,p2fNew,F,(2048,2048))

        dst11 = cv2.warpPerspective(dst1,rectmat1,(2048,2048))
        dst22 = cv2.warpPerspective(dst2,rectmat2,(2048,2048))

        #calculation of the disparity
        stereo = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,ndisparities=16*10, SADWindowSize=9)
        disp = stereo.compute(dst22.astype(uint8), dst11.astype(uint8)).astype(np.float32)
        plt.imshow(disp);plt.colorbar();plt.clim(0,400)#;plt.show()
        plt.savefig("0gauche.png")

        #plot depth by using disparity focal length C1[0,0] from stereo calibration and T[0] the distance between cameras

        plt.imshow(C1[0,0]*T[0]/(disp),cmap='hot');plt.clim(-0,500);plt.colorbar();plt.show()
        '''

if __name__=='__main__':
    #video2frame_quad('video2/2.mkv')
    #video2frame('calibration', [540, 960])
    slice_img('calibration/ZED')
