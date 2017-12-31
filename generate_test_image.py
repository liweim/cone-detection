import cv2
import numpy as np

img = cv2.imread('test.png')
img = cv2.resize(img,(25,25))
patch_radius = 12
pad_width = ((patch_radius, patch_radius), (patch_radius, patch_radius), (0, 0))
img_pad = np.lib.pad(img, pad_width, 'mean')
# dst = np.zeros((50,50,3)).astype(np.uint8)
# dst[12:37,12:37] = img
# cv2.imshow('dst',dst)
# cv2.waitKey(0)
cv2.imwrite('test_image.png', img_pad)
