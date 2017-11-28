import cv2
import numpy as np
import imutils

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

if __name__=='__main__':
	img = cv2.imread('1.png')
	row, col, n = img.shape
	boxes = [[172,302,191,326],[88,300,100,318],[1,304,8,320],[11,287,20,303],[96,284,105,302],[180,283,191,302],[251,286,262,304],[379,288,391,310],[602,323,631,360]]
	boxes = np.array(boxes)
	ymins = boxes[:,0]
	xmins = boxes[:,1]
	ymaxs = boxes[:,2]
	xmaxs = boxes[:,3]

	for xmin, ymin, xmax, ymax in zip(xmins, ymins, xmaxs, ymaxs):
		roi = img[xmin:xmax, ymin:ymax, :]
		blur = cv2.GaussianBlur(roi, (5, 5), 0)
		gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

		hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
		h = hsv[:,:,0]
		s = hsv[:,:,1]
		v = hsv[:,:,2]

		median = 1.2 * np.median(v)#h for blue, v for yellow
		row, col = v.shape
		thresh = np.copy(v)
		thresh[:] = 0
		result = np.copy(thresh)
		for r in range(row):
		    for c in range(col):
		        if v[r, c] > median:
		            thresh[r, c] = 255
		        else:
		            thresh[r, c] = 0
		#edges = cv2.Canny(h,100,200)
		#thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,5)
		#thresh = 255 - thresh

		#kernel = np.ones((5,5), np.uint8)
		#img_erosion = cv2.erode(img, kernel, iterations=1)
		#dilate = cv2.dilate(thresh, kernel, iterations=1)
		#closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

		'''
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
		'''

		cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
		cv2.imshow('gray', gray)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

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
		result[-1, :] = 0
		result[:, 0] = 0
		result[:, -1] = 0
		cv2.namedWindow('result', cv2.WINDOW_NORMAL)
		cv2.imshow('result', result)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

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
			    cX = int((M["m10"] / M["m00"]))
			    cY = int((M["m01"] / M["m00"]))
			except ZeroDivisionError:
			    continue
			#print(cX, cY, area)

			shape = sd.detect(c)

			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the img
			if shape == 'triangle':
			    cv2.drawContours(roi, [c], -1, (0, 255, 0), 2)
			    cv2.putText(roi, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
			    	0.5, (255, 255, 255), 2)

		# show the output img
		cv2.imshow("img", roi)
		cv2.waitKey(0)
