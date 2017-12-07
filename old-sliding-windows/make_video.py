from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
import numpy as np

def make_video(img_path, save_path, fps=1.5, size=None, is_color=True, format='XVID'):
    images_path = os.listdir(img_path)
    images = []
    for image_path in images_path:
        images.append(int(image_path[:-4]))
    images = np.sort(images)

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        image = os.path.join(img_path,str(image)+'.png')
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(save_path, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()

if __name__=='__main__':
    make_video('result/2_5_right/','result/2_5_right.avi' )
