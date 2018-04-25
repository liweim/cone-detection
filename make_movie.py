import skvideo.io
import numpy as np
import os
import cv2

# image_folder = 'annotations/Revolve/results/image'
# writer = skvideo.io.FFmpegWriter("annotations/Revolve/result.mp4", 
# 	inputdict={"-r": "10"}, 
# 	outputdict={'-vcodec': 'libx264', '-b': '30000000'})
# for i in range(1,368):
# 	img_path = image_folder+str(i)+'.png'
# 	if os.path.exists(img_path):
# 		img = cv2.imread(img_path)
# 		writer.writeFrame(img[:,:,::-1])
# writer.close()

image_folder = 'annotations/snowy/results_best/'
writer = skvideo.io.FFmpegWriter("tmp/results/snowy.mp4", 
	inputdict={"-r": "8"}, 
	outputdict={'-vcodec': 'libx264', '-b': '30000000'})
for i in range(0,319):
	img_path = image_folder+str(i)+'.png'
	if os.path.exists(img_path):
		img = cv2.imread(img_path)
		writer.writeFrame(img[:,:,::-1])
writer.close()
