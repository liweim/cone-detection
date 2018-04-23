import cv2
import glob

img_folder_path = 'tmp/images'

for img_path in glob.glob(img_folder_path + '/*.png'):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (320,180))
    img = img[80:140, :]
    cv2.imwrite(img_path, img)
