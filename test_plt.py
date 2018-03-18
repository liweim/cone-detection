import matplotlib.pyplot as plt
import cv2

img = cv2.imread('annotations/skidpad1/1513418735.872856.png')
# plt.ion()
plt.figure(num='astronaut',figsize=(8,8))
plt.imshow(img)
plt.show()