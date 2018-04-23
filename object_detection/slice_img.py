from skimage.io import imread, imsave
import numpy as np
import os
from os.path import join
from shutil import copyfile, rmtree

def read_img(img_path,type=np.uint8):
    img = imread(img_path).astype(type)
    if  len(img.shape)==3 and img.shape[2] == 4:
        img = np.ascontiguousarray(img[:,:,0:3])
    return img

def slice_img(img_path, img_size):
    dirname = os.path.split(img_path)[0]
    num = os.path.split(dirname)[1]
    save_path=join('tmp',num,'images')
    if os.path.exists(save_path):
        rmtree(save_path)
    img=read_img(img_path)
    r,c,n=img.shape
    rows = int(r/img_size)
    cols = int(c/img_size)
    rl=int(np.floor(r/rows))
    cl=int(np.floor(c/cols))
    n=0

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in range(0,r,rl):
        for j in range(0,c,cl):
            if i+rl-1<r and j+cl-1<c:
                n=n+1
                if len(img.shape)==3:
                    temp=img[i:i+rl-1,j:j+cl-1,:]
                else:
                    temp=img[i:i+rl-1,j:j+cl-1]
                if np.min(temp)<255 and np.max(temp)>0:
                    save_img_path = join(save_path, num+'_'+str(n)+'.png')
                    imsave(save_img_path,temp)
    return

if __name__ == '__main__':
    slice_img('tmp/995/995.tiff', 300)
