import skimage.io
import numpy as np
import cv2

def read_img(path_to_image):
    img = skimage.io.imread(path_to_image).astype(np.uint8)
    if img.shape[2] > 3:
        img = np.ascontiguousarray(img[:,:,0:3])

    if np.max(img) > 255.0:
        raise Exception('Only 8 bit images supported')
    return img[:,:,::-1]

# def write_txt(txt,point,way='w'):
#     with open(txt,way) as f:
#         for i in range(len(point)):
#             f.write(str(point[i,0])+' '+str(point[i,1]))
#             f.write('\n')

def write_txt(txt_path, data, way='w'):
    with open(txt_path, way) as f:
        for d in data:
            for i in range(len(d)):
                f.write(str(d[i])+' ')
            f.write('\n')

# def read_txt(txt_path):
#     with open(txt_path, 'r') as f:
#         s=f.readlines()
#         point=np.zeros((len(s),2))
#         for i in range(len(s)):
#             arr=s[i].split(' ')
#             point[i,:]=arr
#     return point

def read_txt(txt_path):
    data = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data.append(line.split(' '))
    return data
