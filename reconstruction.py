import numpy as np
import cv2
import os
from os.path import join
import matplotlib.pyplot as plt
import pylab as pl
import time
from rectify import rectify
import argparse

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
F = 700
d = 120

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')


def reconstruction(img_path):
    imgL, imgR = rectify(img_path)
    #plt.imshow((imgL+imgR)[:,:,::-1])
    #plt.show()
    #imgL = cv2.pyrDown(imgL)
    #imgR = cv2.pyrDown(imgR)

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 8,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    '''
    f, ax = plt.subplots(2, 2)
    ax[0][0].imshow(imgL[:,:,::-1])
    ax[0][1].imshow(imgR[:,:,::-1])
    ax[1][0].imshow((imgL+imgR)[:,:,::-1])
    ax[1][1].imshow(disp, cmap='gray')
    plt.show()
    '''

    factor = F * d / 4
    return imgL, imgR, disp, factor

    '''
    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -F], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)
    colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
    out_fn = 'out.ply'
    write_ply('out.ply', out_points, out_colors)
    print('%s saved' % 'out.ply')
    '''

    '''
    xmin = int(points[:, :, 0].min() * 10)
    xmax = int(points[:, :, 0].max() * 10) - xmin + 1
    ymin = int(points[:, :, 2].min() * 10)
    ymax = int(points[:, :, 2].max() * 10) - ymin + 1
    print(xmin, xmax, ymin, ymax)
    plane = np.zeros([ymax, xmax, 3])
    for i in range(h):
        for j in range(w):
            if disp[i, j] > disp.min():
                x = int(points[i, j, 0] * 10) - xmin
                y = int(points[i, j, 2] * 10) - ymin
                plane[y, x] = colors[i, j]
    plane = cv2.flip(plane, 0)
    plane = cv2.flip(plane, 1)
    plt.imshow(plane)
    plt.show()
    '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    args = parser.parse_args()

    reconstruction(img_path = args.img_path)
