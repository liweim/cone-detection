import cv2
import numpy as np
import imutils
import os
from os.path import join
import pandas as pd
import random

class ShapeDetector:
	def __init__(self):
		pass

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "0"
		peri = 0.06 * cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, peri, True)#0.04
        		# if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"

		return shape

def detect_cone(img, roi, bias, label, color, lower_hsv, higher_hsv):
	row, col = roi.shape
	result = np.zeros(roi.shape[:2], dtype = np.uint8)
	thresh = roi
	'''
	cv2.namedWindow('roi', cv2.WINDOW_NORMAL)
	cv2.imshow('roi', roi)
	cv2.waitKey(0)
	cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
	cv2.imshow('thresh', thresh)
	cv2.waitKey(0)
	'''
	cX = int(row/2)
	cY = int(col/2)
	result[cX, cY] = 255
	neighbor=[[cX, cY]]
	num=len(neighbor)
	while 1:
		old_neighbor=neighbor
		iter=len(old_neighbor)
		for j in range(iter):
			x_=old_neighbor[j][0]
			y_=old_neighbor[j][1]
			if x_-1>=0 and thresh[x_-1,y_]>0 and [x_-1,y_] not in neighbor:
				neighbor.append([x_-1,y_])
				result[x_-1,y_] = 255
			if x_+1<row and thresh[x_+1,y_]>0 and [x_+1,y_] not in neighbor:
				neighbor.append([x_+1,y_])
				result[x_+1,y_] = 255
			if y_-1>=0 and thresh[x_,y_-1]>0 and [x_,y_-1] not in neighbor:
				neighbor.append([x_,y_-1])
				result[x_,y_-1] = 255
			if y_+1<col and thresh[x_,y_+1]>0 and [x_,y_+1] not in neighbor:
				neighbor.append([x_,y_+1])
				result[x_,y_+1] = 255
		if not len(neighbor)==num:
			num=len(neighbor)
		else:
			break
	result[0, :] = 0
	result[int(row * 0.75):-1, :] = 0
	result[-1, :] = 0
	result[:, 0] = 0
	result[:, -1] = 0
	'''
	cv2.namedWindow('detect: '+color, cv2.WINDOW_NORMAL)
	cv2.imshow('detect: '+color, result)
	cv2.waitKey(0)
	'''
	cnts = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	sd = ShapeDetector()
	# loop over the contours
	for c in cnts:
		area = cv2.contourArea(c)
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)
		try:
			cX = int((M["m10"] / M["m00"]))+bias[0]
			cY = int((M["m01"] / M["m00"]))+bias[1]
		except ZeroDivisionError:
			continue
		shape = sd.detect(c)

		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the img
		if shape == 'triangle':
			if color == 'yellow':
				cv2.drawContours(img, [c+bias], -1, (0, 255, 255), 2)
			if color == 'blue':
				cv2.drawContours(img, [c+bias], -1, (255, 0, 0), 2)
			cv2.putText(img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (255, 255, 255), 2)

	return result

def callback(x):
	pass

def feature_based(bbox_folder_path):
	dirname = os.path.split(bbox_folder_path)[0]
	bbox_paths = os.listdir(bbox_folder_path)


	for i in range(len(bbox_paths)):
		bbox_path = random.choice(bbox_paths)
		filename = os.path.splitext(bbox_path)[0]
		img_path = join(dirname, 'right', filename+'.png')
		img = cv2.imread(img_path)
		print(img_path)
		row, col, n = img.shape
		blur = cv2.GaussianBlur(img, (5, 5), 0)
		hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

		lower_hsv = np.array([0, 0, 60])
		higher_hsv = np.array([100, 255, 255])
		yellow = cv2.inRange(hsv, lower_hsv, higher_hsv)

		lower_hsv = np.array([105, 120, 0])
		higher_hsv = np.array([120, 255, 255])
		blue = cv2.inRange(hsv, lower_hsv, higher_hsv)

		bbox_df = pd.read_csv(join(bbox_folder_path, bbox_path))
		boxes = np.array(bbox_df[:])
		labels = boxes[:,0]
		ymins = boxes[:,1]
		xmins = boxes[:,2]
		ymaxs = boxes[:,3]
		xmaxs = boxes[:,4]
		cxs = (xmins+xmaxs)/2
		cys = (ymins+ymaxs)/2

		for label, xmin, ymin, xmax, ymax, cx, cy in zip(labels, xmins, ymins, xmaxs, ymaxs, cxs, cys):
			bias = [int(ymin), int(xmin)]

			roi = yellow[xmin:xmax, ymin:ymax]
			detect_cone(img, roi, bias, label, 'yellow', lower_hsv, higher_hsv)

			roi = blue[xmin:xmax, ymin:ymax]
			detect_cone(img, roi, bias, label, 'blue', lower_hsv, higher_hsv)

		cv2.namedWindow('img', cv2.WINDOW_NORMAL)
		cv2.imshow('img', img)
		cv2.namedWindow('yellow', cv2.WINDOW_NORMAL)
		cv2.imshow('yellow', yellow)
		cv2.namedWindow('blue', cv2.WINDOW_NORMAL)
		cv2.imshow('blue', blue)
		cv2.waitKey(0)

	cv2.destroyAllWindows()

if __name__=='__main__':
	feature_based('video2/bbox')
