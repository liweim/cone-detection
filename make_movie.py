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

# image_folder = 'annotations/snowy/results_best/'
# writer = skvideo.io.FFmpegWriter("tmp/results/snowy.mp4",
# 	inputdict={"-r": "8"},
# 	outputdict={'-vcodec': 'libx264', '-b': '30000000'})
# for i in range(0,319):
# 	img_path = image_folder+str(i)+'.png'
# 	if os.path.exists(img_path):
# 		img = cv2.imread(img_path)
# 		writer.writeFrame(img[:,:,::-1])
# writer.close()

# image_folder1 = 'tmp/sd_results/'
# image_folder2 = 'tmp/svo_results/'
# writer = skvideo.io.FFmpegWriter("tmp/results/result.mp4",
# 	inputdict={"-r": "8"},
# 	outputdict={'-vcodec': 'libx264', '-b': '30000000'})
# for i in range(0,224):
# 	img1 = cv2.imread(image_folder1+str(i+87)+'.png')
# 	img2 = cv2.imread(image_folder2+str(i)+'.png')
# 	img2 = cv2.resize(img2,(376,376));
# 	img = np.hstack((img1,img2));
# 	writer.writeFrame(img[:,:,::-1])
# writer.close()

# image_folder = 'data/hairpin/results/'
# writer = skvideo.io.FFmpegWriter("tmp/results/hairpin.mp4",
# 	inputdict={"-r": "15"},
# 	outputdict={'-vcodec': 'libx264', '-b': '30000000'})
# for i in range(0,805):
# 	img = cv2.imread(image_folder+str(i)+'.png')
# 	writer.writeFrame(img[:,:,::-1])
# writer.close()

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('tmp/results/hairpin.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (1280,720))
for i in range(0,805):
	img_path = 'data/hairpin/results/'+str(i)+'.png'
	img = cv2.imread(img_path)
	out.write(img)
out.release()
