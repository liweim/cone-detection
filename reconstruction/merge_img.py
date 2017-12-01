import cv2

left = cv2.imread('result/6_left.png')
right = cv2.imread('result/6_right.png')

cv2.namedWindow('merge', cv2.WINDOW_NORMAL)
cv2.imshow('merge', left+right)
cv2.waitKey(0)
