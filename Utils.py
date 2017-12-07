import skimage.io
import numpy as np
import cv2

def load_std_image(path_to_image):
    img = skimage.io.imread(path_to_image).astype(np.float32)
    img = img/255
    if img.shape[2] == 4:
        img = np.ascontiguousarray(img[:,:,0:3])

    if np.max(img) > 1.0:
        raise Exception('Only 8 bit images supported')
    return img

def read_img(img_path,type=np.uint8):
    img = skimage.io.imread(img_path).astype(type)
    if 0:
        img2=np.zeros([img.shape[0],img.shape[1],3],dtype=np.uint8)
        img2[:,:,0]=img2[:,:,1]=img2[:,:,2]=img[:,:]
    if  len(img.shape)==3 and img.shape[2] == 4:
        img = np.ascontiguousarray(img[:,:,0:3])
    return img

def write_txt(txt,point,way='w'):
    with open(txt,way) as f:
        for i in range(len(point)):
            if point[i,0]!=0 and point[i,1]!=0:
                f.write(str(point[i,0])+' '+str(point[i,1]))
                f.write('\n')

def read_txt(txt):
    with open(txt,'r') as f:
        s=f.readlines()
        point=np.zeros((len(s),2))
        for i in range(len(s)):
            arr=s[i].split(' ')
            point[i,:]=arr
    return point
