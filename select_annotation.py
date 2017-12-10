import numpy as np
import os
from os.path import join
from Utils import read_txt, write_txt
import matplotlib.pyplot as plt
import pylab as pl
from rename_img import rename_img
import cv2
import argparse

patch_size = 25

def getPolygonBounds(points):
    length = len(points)
    top = down = left = right = points[0]
    for i in range(1, length):
        if points[i][0] > top[0]:
            top = points[i]
        elif points[i][0] < down[0]:
            down = points[i]
        else:
            pass
        if points[i][1] > right[1]:
            right = points[i]
        elif points[i][1] < left[1]:
            left = points[i]
        else:
            pass

    point0 = [top[0], left[1]]
    point1 = [top[0], right[1]]
    point2 = [down[0], right[1]]
    point3 = [down[0], left[1]]
    polygonBounds = [point0, point1, point2, point3]
    return polygonBounds

def isPointInRect(point, polygonBounds):
    if point[0] >= polygonBounds[3][0] and point[0] <= polygonBounds[0][0] and point[1] >= polygonBounds[3][1] and point[1] <= polygonBounds[2][1]:
        return True
    else:
        return False

def isPointsInPolygons(xyset,polygonset):
    inpolygonsetxyList = []
    for points in polygonset:
        polygonBounds = getPolygonBounds(points)
        for point in xyset:
            if not isPointInRect(point, polygonBounds):
                continue

            length = len(points)
            p = point
            p1 = points[0]
            flag = False
            for i in range(1, length):
                p2 = points[i]
                if (p[0] == p1[0] and p[1] == p1[1]) or (p[0] == p2[0] and p[1] == p2[1]):
                    inpolygonsetxyList.append(p)
                    break
                if (p2[1] < p[1] and p1[1] >= p[1]) or (p2[1] >= p[1] and p1[1] < p[1]):
                    if (p2[1] == p1[1]):
                        x = (p1[0] + p2[0])/2
                    else:
                        x = p2[0] - (p2[1] - p[1])*(p2[0] - p1[0])/(p2[1] - p1[1])
                    if (x == p[0]):
                        inpolygonsetxyList.append(p)
                        break
                    if (x > p[0]):
                        flag = not flag
                    else:
                        pass
                else:
                    pass

                p1 = p2
            if flag:
                inpolygonsetxyList.append(p)
    return inpolygonsetxyList

def select_annotation(model_id, plant_id, img_path, radius):
    threshold = 500
    img_path = plant_id + '_' + img_path
    mark_path = join('tmp', plant_id, img_path)
    filename = os.path.splitext(mark_path)[0]
    img = cv2.imread(mark_path)
    source_path = join('tmp', plant_id, plant_id, img_path)
    img_source = cv2.imread(source_path)

    txt_path = filename + '.txt'
    idx = read_txt(txt_path)
    plt.imshow(img[:,:,::-1])
    point = []
    point.append(np.round(pl.ginput(1, timeout = 10^10)))
    i = 0
    flag = 1
    plt.scatter(point[i][0][0], point[i][0][1], marker='o', color='b', s=10)
    while flag:
        i += 1
        point.append(np.round(pl.ginput(1, timeout = 10^10)))
        if (point[i][0][0] - point[0][0][0])**2 + (point[i][0][1] - point[0][0][1])**2 < threshold:
            flag=0
        else:
            plt.scatter(point[i][0][0], point[i][0][1], marker='o', color='b', s=10)
            plt.plot([point[i][0][0], point[i-1][0][0]], [point[i][0][1], point[i-1][0][1]], 'b')
    point[i] = point[0]
    plt.plot([point[i][0][0], point[i-1][0][0]], [point[i][0][1], point[i-1][0][1]], 'b')

    xy = []
    poly = []
    polygonset = []
    for i in range(len(idx)):
        xy.append([float(idx[i][1]), float(idx[i][0])])
    for i in range(len(point)):
        poly.append([float(point[i][0][0]), float(point[i][0][1])])
    polygonset.append(poly)

    outliers = isPointsInPolygons(xy, polygonset)
    outliers = np.array(outliers)
    for i in range(len(outliers)):
        plt.scatter(outliers[i][0], outliers[i][1], marker='x', color='b', s=30)
    plt.show()

    n = len(os.listdir(join('images', model_id)))
    mask = np.zeros(img.shape[:2], dtype = np.uint8)

    for i in range(outliers.shape[0]):
        #point.dtype='int64'
        x = int(outliers[i, 0])
        y = int(outliers[i, 1])
        cv2.circle(mask, (x, y), radius, 100, -1)

    poly = np.array(poly)
    xmin = int(min(poly[:,1]))
    xmax = int(max(poly[:,1]))
    ymin = int(min(poly[:,0]))
    ymax = int(max(poly[:,0]))
    img_roi = img_source[xmin:xmax, ymin:ymax]
    mask_roi = mask[xmin:xmax, ymin:ymax]
    cv2.imwrite(join('annotations', model_id, plant_id + '_' + str(n+1) + '.png'), mask_roi)
    cv2.imwrite(join('images', model_id, plant_id + '_' + str(n+1) + '.png'), img_roi)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, help="Which plant to detect.")
    parser.add_argument("--plant_id", type=str, help="Plant id.")
    parser.add_argument("--img_path", type=str, help="Image to analyze.")
    parser.add_argument("--radius", type=int, help="Annotation radius.")
    args = parser.parse_args()

    select_annotation(model_id = args.model_id, plant_id = args.plant_id, img_path = args.img_path, radius = args.radius)
