import cv2
import numpy as np
import os
from os.path import join
import argparse

def callback(x):
    pass

def select_region(cone_id, img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    row, col = img.shape[:2]

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    r = cv2.selectROI('mask', img)
    if r == (0,0,0,0):
        img_roi = img
        hsv_roi = hsv
    else:
        rl = max(int(r[1]), 0)
        rr = min(int(r[1]+r[3]), row)
        cl = max(int(r[0]), 0)
        cr = min(int(r[0]+r[2]), col)
        img_roi = img[rl:rr, cl:cr]
        hsv_roi = hsv[rl:rr, cl:cr]

    print('select region')
    cv2.createTrackbar('h_low','mask',10,179,callback)
    cv2.createTrackbar('h_high','mask',100,179,callback)

    cv2.createTrackbar('s_low','mask',0,255,callback)
    cv2.createTrackbar('s_high','mask',255,255,callback)

    cv2.createTrackbar('v_low','mask',0,255,callback)
    cv2.createTrackbar('v_high','mask',255,255,callback)

    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        h_low = cv2.getTrackbarPos('h_low','mask')
        h_high = cv2.getTrackbarPos('h_high','mask')
        s_low = cv2.getTrackbarPos('s_low','mask')
        s_high = cv2.getTrackbarPos('s_high','mask')
        v_low = cv2.getTrackbarPos('v_low','mask')
        v_high = cv2.getTrackbarPos('v_high','mask')

        lower_hsv = np.array([h_low, s_low, v_low])
        higher_hsv = np.array([h_high, s_high, v_high])

        mask = cv2.inRange(hsv_roi, lower_hsv, higher_hsv)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask_roi = cv2.bitwise_and(img_roi, img_roi, mask = mask)
        result = cv2.hconcat((mask_roi, img_roi))

        cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
        cv2.imshow('mask', result)

    row, col = img_roi.shape[:2]
    r = cv2.selectROI('mask', mask_roi)
    if not r == (0,0,0,0):
        rl = max(int(r[1]), 0)
        rr = min(int(r[1]+r[3]), row)
        cl = max(int(r[0]), 0)
        cr = min(int(r[0]+r[2]), col)
        img_roi = img_roi[rl:rr, cl:cr]
        mask_roi = mask[rl:rr, cl:cr]
    cv2.destroyAllWindows()

    row, col = mask_roi.shape
    for r in range(row):
        for c in range(col):
            if mask_roi[r, c] > 0:
                mask_roi[r, c] = 255
            else:
                mask_roi[r, c] = 100
    n = len(os.listdir(join('images', cone_id)))
    basename = cone_id+'_'+str(n+1)+'.png'
    cv2.imwrite(join('images', cone_id, basename), img_roi)
    cv2.imwrite(join('annotations', cone_id, basename), mask_roi)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cone_id", type=str, help="Which plant to detect.")
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    args = parser.parse_args()

    select_region(cone_id = args.cone_id, img_path = args.img_path)
