import os
from os.path import join
from rectify import rectify
import argparse

def rectify_all(img_folder_path):
    img_paths = os.listdir(img_folder_path)
    for img_path in img_paths:
        rectify(join(img_folder_path, img_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder_path", type=str, help="Image to analyze.")
    args = parser.parse_args()

    rectify_all(img_folder_path = args.img_folder_path)
