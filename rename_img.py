import numpy as np
import os
from os.path import join
import argparse

def rename_img(img_path):
    filenames = os.listdir(img_path)
    for i in range(len(filenames)):
        ext = os.path.splitext(filenames[i])[1]
        os.rename(join(img_path, filenames[i]), join(img_path, str(i + 1) + ext + 't'))

    filenames = os.listdir(img_path)
    for i in range(len(filenames)):
        os.rename(join(img_path, filenames[i]), join(img_path, str(i + 1) + '.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type = str)
    args = parser.parse_args()

    rename_img(img_path = args.img_path)
