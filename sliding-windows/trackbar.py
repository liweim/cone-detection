import cv2
import numpy as np

def callback(x):
    pass

def trackbar(bbox_folder_path):
    dirname = os.path.split(bbox_folder_path)[0]
    bbox_paths = os.listdir(bbox_folder_path)
    bbox_path = random.choice(bbox_paths)
    filename = os.path.splitext(bbox_path)[0]
    img_path = join(dirname, 'right', filename+'.png')

    img = cv2.imread(img_path)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img',img)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)

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
        mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
        cv2.imshow('mask',mask)

    cv2.destroyAllWindows()

if __name__=='__main__':
	trackbar('video2/bbox')
