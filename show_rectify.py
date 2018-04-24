import numpy as np
import cv2
import matplotlib.pyplot as plt
import pylab as pl

imgL = cv2.imread('tmp/imgL.png');
imgR = cv2.imread('tmp/imgR.png');
img = imgL+imgR
gray = cv2.cvtColor(img, 6);
plt.imshow(gray, cmap='gray')
plt.show()