import glob
import gc
import numpy as np
import os
from os.path import join
import time
import argparse
from efficient_sliding_window import load_network, cone_detect

def cone_detect_all(img_folder_path, model_path, cone_distance, threshold):
    model = load_network(model_path)
    start = time.clock()
    img_paths = glob.glob(img_folder_path + '/*.png')
    for img_path in img_paths:
        cone_detect(img_path, model, cone_distance, threshold, display_result = 0)
        gc.collect()
    print('Run time: {}'.format((time.clock() - start)/len(img_paths)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder_path", type = str)
    parser.add_argument("--model_path", type = str)
    parser.add_argument("--cone_distance", type = int)
    parser.add_argument("--threshold", type = float)
    args = parser.parse_args()

    cone_detect_all(img_folder_path = args.img_folder_path, model_path = args.model_path, cone_distance = args.cone_distance, threshold = args.threshold)
